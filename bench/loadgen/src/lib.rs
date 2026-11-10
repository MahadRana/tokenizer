//! Open-loop load generator for `POST /v1/chat/completions`.
//!
//! Arrivals are a Poisson process: the gap before each request after the
//! first is an exponential draw with mean `1/rate`. Tasks are spawned and
//! do not wait for the previous request to finish, so a slow server shows
//! up as queueing delay in TTFT instead of being hidden by a closed loop.
//!
//! TTFT is the time from sending the request to the first SSE `data:` line
//! that carries an output token. The chat server emits a role `data:` line
//! as soon as the stream is polled, before prefill returns, so that line is
//! not time-to-first-token. An output token is a `data:` payload with a
//! non-empty `delta.content`, or a non-JSON payload that is not `[DONE]`
//! (a mock server can stay small). Inter-token latency is the gap between
//! those tokens.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use hdrhistogram::Histogram;
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand_distr::Exp;
use reqwest::Client;

/// Gaps, in seconds, before requests `1..count`. Request 0 is at t = 0.
///
/// `rate_per_sec` is the Poisson rate. The mean of the returned gaps is
/// `1/rate`. The same `StdRng` seed reproduces the same gaps.
pub fn poisson_gaps(rng: &mut StdRng, rate_per_sec: f64, count: usize) -> Result<Vec<f64>, String> {
    if !(rate_per_sec.is_finite() && rate_per_sec > 0.0) {
        return Err(format!(
            "rate must be a positive finite number, got {rate_per_sec}"
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }
    let dist = Exp::new(rate_per_sec).map_err(|err| err.to_string())?;
    let mut gaps = Vec::with_capacity(count.saturating_sub(1));
    for _ in 1..count {
        gaps.push(dist.sample(rng));
    }
    Ok(gaps)
}

/// Record latencies in microseconds. Bounds are 1 µs to 10 minutes, 3 sig figs.
pub struct LatencyHistogram {
    inner: Histogram<u64>,
}

impl LatencyHistogram {
    pub fn new() -> Self {
        Self {
            inner: Histogram::<u64>::new_with_bounds(1, 600_000_000, 3).expect("histogram bounds"),
        }
    }

    pub fn record(&mut self, latency: Duration) {
        let us = latency.as_micros().clamp(1, 600_000_000) as u64;
        // Values inside the configured bounds do not fail `record`.
        let _ = self.inner.record(us);
    }

    pub fn len(&self) -> u64 {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// `None` when nothing was recorded. Quantiles are 0.0–1.0.
    pub fn quantile_us(&self, q: f64) -> Option<u64> {
        if self.inner.is_empty() {
            return None;
        }
        Some(self.inner.value_at_quantile(q))
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

/// One SSE line, applied to a request that started at `started`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseObservation {
    pub first_data: Option<Duration>,
    pub token_gaps: Vec<Duration>,
    pub output_tokens: u64,
}

#[derive(Debug)]
struct SseCursor {
    started: Instant,
    first_data: Option<Instant>,
    last_token: Option<Instant>,
    token_gaps: Vec<Duration>,
    output_tokens: u64,
}

impl SseCursor {
    fn new(started: Instant) -> Self {
        Self {
            started,
            first_data: None,
            last_token: None,
            token_gaps: Vec::new(),
            output_tokens: 0,
        }
    }

    fn push_line(&mut self, line: &str, now: Instant) {
        let Some(payload) = data_payload(line) else {
            return;
        };
        if payload == "[DONE]" || !is_output_token(payload) {
            return;
        }
        if self.first_data.is_none() {
            self.first_data = Some(now);
        }
        if let Some(prev) = self.last_token {
            self.token_gaps.push(now.saturating_duration_since(prev));
        }
        self.last_token = Some(now);
        self.output_tokens += 1;
    }

    fn finish(self) -> SseObservation {
        SseObservation {
            first_data: self
                .first_data
                .map(|t| t.saturating_duration_since(self.started)),
            token_gaps: self.token_gaps,
            output_tokens: self.output_tokens,
        }
    }
}

/// `data:` payload, without the prefix. Comment and field-less lines are ignored.
pub fn data_payload(line: &str) -> Option<&str> {
    let rest = line.trim().strip_prefix("data:")?;
    Some(rest.trim())
}

fn is_output_token(payload: &str) -> bool {
    if payload.is_empty() || payload == "[DONE]" {
        return false;
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) else {
        // A mock can send a bare token instead of a chat chunk.
        return true;
    };
    value
        .pointer("/choices/0/delta/content")
        .and_then(|v| v.as_str())
        .is_some_and(|text| !text.is_empty())
}

/// Read an SSE body. `started` is when the request was sent.
pub async fn read_response(response: reqwest::Response, started: Instant) -> SseObservation {
    let mut stream = response.bytes_stream();
    let mut cursor = SseCursor::new(started);
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        let Ok(chunk) = chunk else {
            break;
        };
        buf.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(nl) = buf.find('\n') {
            let line: String = buf.drain(..=nl).collect();
            cursor.push_line(&line, Instant::now());
        }
    }
    if !buf.is_empty() {
        cursor.push_line(&buf, Instant::now());
    }
    cursor.finish()
}

/// One finished request.
#[derive(Debug)]
pub struct Sample {
    pub ttft: Option<Duration>,
    pub e2e: Duration,
    pub token_gaps: Vec<Duration>,
    pub output_tokens: u64,
    pub error: Option<String>,
}

/// What one load run reports. Times are microseconds.
#[derive(Debug, Clone)]
pub struct Report {
    pub requests: usize,
    pub ok: usize,
    pub errors: usize,
    pub output_tokens: u64,
    pub wall_secs: f64,
    pub ttft_p50_us: Option<u64>,
    pub ttft_p95_us: Option<u64>,
    pub ttft_p99_us: Option<u64>,
    pub itl_p50_us: Option<u64>,
    pub itl_p99_us: Option<u64>,
    pub e2e_p99_us: Option<u64>,
    pub output_tokens_per_sec: f64,
    pub requests_per_sec: f64,
}

pub fn summarize(samples: &[Sample], wall: Duration) -> Report {
    let mut ttft = LatencyHistogram::new();
    let mut itl = LatencyHistogram::new();
    let mut e2e = LatencyHistogram::new();
    let mut output_tokens = 0u64;
    let mut errors = 0usize;
    for sample in samples {
        if sample.error.is_some() {
            errors += 1;
            continue;
        }
        if let Some(latency) = sample.ttft {
            ttft.record(latency);
        }
        for gap in &sample.token_gaps {
            itl.record(*gap);
        }
        e2e.record(sample.e2e);
        output_tokens += sample.output_tokens;
    }
    let wall_secs = wall.as_secs_f64().max(1e-9);
    let ok = samples.len().saturating_sub(errors);
    Report {
        requests: samples.len(),
        ok,
        errors,
        output_tokens,
        wall_secs,
        ttft_p50_us: ttft.quantile_us(0.50),
        ttft_p95_us: ttft.quantile_us(0.95),
        ttft_p99_us: ttft.quantile_us(0.99),
        itl_p50_us: itl.quantile_us(0.50),
        itl_p99_us: itl.quantile_us(0.99),
        e2e_p99_us: e2e.quantile_us(0.99),
        output_tokens_per_sec: output_tokens as f64 / wall_secs,
        requests_per_sec: ok as f64 / wall_secs,
    }
}

#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub url: String,
    pub requests: usize,
    pub rate_per_sec: f64,
    pub seed: u64,
    pub prompt: String,
    pub max_tokens: u32,
    pub model: String,
    /// Requests issued before the timed run. Their samples are dropped.
    pub warmup: usize,
}

/// Open-loop run. Each request is its own task. The spawn moves owned data
/// in (`async move`) because `tokio::spawn` requires `'static + Send`.
pub async fn run_load(config: LoadConfig) -> Result<Report, String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .map_err(|err| err.to_string())?;
    if config.warmup > 0 {
        let warm = one_request(
            &client,
            &config.url,
            &config.model,
            &config.prompt,
            config.max_tokens,
        )
        .await;
        if let Some(err) = warm.error {
            return Err(format!("warmup failed: {err}"));
        }
    }
    let mut rng = StdRng::seed_from_u64(config.seed);
    let gaps = poisson_gaps(&mut rng, config.rate_per_sec, config.requests)?;
    let (tx, mut rx) = tokio::sync::mpsc::channel(config.requests.max(1));
    let origin = tokio::time::Instant::now();
    let wall_start = Instant::now();
    let mut cursor = Duration::ZERO;
    for index in 0..config.requests {
        if index > 0 {
            cursor += Duration::from_secs_f64(gaps[index - 1]);
        }
        let when = origin + cursor;
        let tx = tx.clone();
        let client = client.clone();
        let url = config.url.clone();
        let model = config.model.clone();
        let prompt = config.prompt.clone();
        let max_tokens = config.max_tokens;
        tokio::spawn(async move {
            tokio::time::sleep_until(when).await;
            let sample = one_request(&client, &url, &model, &prompt, max_tokens).await;
            let _ = tx.send(sample).await;
        });
    }
    drop(tx);
    let mut samples = Vec::with_capacity(config.requests);
    while let Some(sample) = rx.recv().await {
        samples.push(sample);
    }
    Ok(summarize(&samples, wall_start.elapsed()))
}

async fn one_request(
    client: &Client,
    url: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
) -> Sample {
    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": true,
    });
    let started = Instant::now();
    let response = match client.post(url).json(&body).send().await {
        Ok(response) => response,
        Err(err) => {
            return Sample {
                ttft: None,
                e2e: started.elapsed(),
                token_gaps: Vec::new(),
                output_tokens: 0,
                error: Some(err.to_string()),
            };
        }
    };
    if !response.status().is_success() {
        let status = response.status();
        return Sample {
            ttft: None,
            e2e: started.elapsed(),
            token_gaps: Vec::new(),
            output_tokens: 0,
            error: Some(format!("http {status}")),
        };
    }
    let observed = read_response(response, started).await;
    Sample {
        ttft: observed.first_data,
        e2e: started.elapsed(),
        token_gaps: observed.token_gaps,
        output_tokens: observed.output_tokens,
        error: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn poisson_mean_is_within_tolerance_for_a_fixed_seed() {
        let mut rng = StdRng::seed_from_u64(42);
        let rate = 10.0;
        let gaps = poisson_gaps(&mut rng, rate, 20_001).unwrap();
        assert_eq!(gaps.len(), 20_000);
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let expected = 1.0 / rate;
        assert!(
            (mean - expected).abs() < 0.005,
            "mean {mean} expected {expected}"
        );
        let mut again = StdRng::seed_from_u64(42);
        let replay = poisson_gaps(&mut again, rate, 20_001).unwrap();
        assert_eq!(gaps, replay);
        assert!(poisson_gaps(&mut rng, 0.0, 4).is_err());
    }

    #[test]
    fn histogram_percentiles_match_a_known_sample() {
        let mut hist = LatencyHistogram::new();
        assert!(hist.quantile_us(0.5).is_none());
        for value in 1u64..=100 {
            hist.record(Duration::from_micros(value));
        }
        let p50 = hist.quantile_us(0.50).unwrap();
        let p95 = hist.quantile_us(0.95).unwrap();
        let p99 = hist.quantile_us(0.99).unwrap();
        assert!((p50 as i64 - 50).abs() <= 1, "p50 {p50}");
        assert!((p95 as i64 - 95).abs() <= 2, "p95 {p95}");
        assert!((p99 as i64 - 99).abs() <= 2, "p99 {p99}");

        let samples = [Sample {
            ttft: Some(Duration::from_micros(1_000)),
            e2e: Duration::from_micros(5_000),
            token_gaps: vec![Duration::from_micros(200), Duration::from_micros(400)],
            output_tokens: 3,
            error: None,
        }];
        let report = summarize(&samples, Duration::from_secs(2));
        assert_eq!(report.ok, 1);
        assert_eq!(report.output_tokens, 3);
        assert!((report.output_tokens_per_sec - 1.5).abs() < 1e-9);
        assert_eq!(report.ttft_p50_us, Some(1_000));
        assert!(report.itl_p50_us.is_some());
        let e2e = report.e2e_p99_us.unwrap();
        assert!((e2e as i64 - 5_000).abs() <= 10, "e2e p99 {e2e}");
    }

    #[tokio::test]
    async fn sse_ttft_is_the_first_content_data_event_from_a_mock_server() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut byte = [0u8; 1];
            loop {
                sock.read_exact(&mut byte).await.unwrap();
                buf.push(byte[0]);
                if buf.ends_with(b"\r\n\r\n") || buf.len() > 64 * 1024 {
                    break;
                }
            }
            sock.write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
            )
            .await
            .unwrap();
            tokio::time::sleep(Duration::from_millis(40)).await;
            write_chunk(
                &mut sock,
                "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            )
            .await;
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_chunk(
                &mut sock,
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            )
            .await;
            tokio::time::sleep(Duration::from_millis(30)).await;
            write_chunk(
                &mut sock,
                "data: {\"choices\":[{\"delta\":{\"content\":\"!\"}}]}\n\n",
            )
            .await;
            write_chunk(&mut sock, "data: [DONE]\n\n").await;
            sock.write_all(b"0\r\n\r\n").await.unwrap();
        });

        let started = Instant::now();
        let response = reqwest::Client::new()
            .post(format!("http://{addr}/v1/chat/completions"))
            .json(&serde_json::json!({"stream": true}))
            .send()
            .await
            .unwrap();
        assert!(response.status().is_success());
        let observed = read_response(response, started).await;
        let ttft = observed.first_data.expect("first content data event");
        assert!(
            ttft >= Duration::from_millis(60),
            "ttft {ttft:?} should wait for the content data: line, not the role line"
        );
        assert!(ttft < Duration::from_millis(500), "ttft {ttft:?}");
        assert_eq!(observed.output_tokens, 2);
        assert_eq!(observed.token_gaps.len(), 1);
        assert!(
            observed.token_gaps[0] >= Duration::from_millis(15),
            "inter-token {:?}",
            observed.token_gaps[0]
        );
    }

    async fn write_chunk(sock: &mut tokio::net::TcpStream, data: &str) {
        let header = format!("{:x}\r\n", data.len());
        sock.write_all(header.as_bytes()).await.unwrap();
        sock.write_all(data.as_bytes()).await.unwrap();
        sock.write_all(b"\r\n").await.unwrap();
        sock.flush().await.unwrap();
    }

    #[test]
    fn data_prefix_and_done_are_not_tokens() {
        assert_eq!(data_payload("data: [DONE]\n"), Some("[DONE]"));
        assert_eq!(data_payload(": keep-alive\n"), None);
        assert!(!is_output_token("[DONE]"));
        assert!(!is_output_token(
            r#"{"choices":[{"delta":{"role":"assistant"}}]}"#
        ));
        assert!(is_output_token(
            r#"{"choices":[{"delta":{"content":"Hi"}}]}"#
        ));
    }
}
