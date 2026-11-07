//! Protocol tests against `tiny_model`. The weights are a patterned net, not
//! Qwen. The HTTP shape is what these tests lock down. The ignored test at
//! the bottom loads the real checkpoint and talks to it with the OpenAI SDK.

use std::sync::Arc;
use std::time::Duration;

use qwen_server::{ChatTurn, Server, TokenCodec, apply_qwen_chat_template};
use serde_json::{Value, json};
use tokenizer::TokenizerError;

struct CharCodec;

impl TokenCodec for CharCodec {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        // Ids have to land in the tiny model's vocab (32). One id per scalar
        // so `prompt_tokens` is the char length of the rendered template.
        Ok(text.chars().map(|ch| (c_as_u32(ch) % 30) + 1).collect())
    }

    fn decode_token(&self, id: u32) -> Result<Vec<u8>, TokenizerError> {
        Ok(vec![b'a' + (id % 26) as u8])
    }
}

fn c_as_u32(ch: char) -> u32 {
    u32::from(ch)
}

/// Same ids as [`CharCodec`], but each decode sleeps so a client can drop
/// the socket before a 200-token run fits in the kernel buffer.
struct SlowCodec;

impl TokenCodec for SlowCodec {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        TokenCodec::encode(&CharCodec, text)
    }

    fn decode_token(&self, id: u32) -> Result<Vec<u8>, TokenizerError> {
        std::thread::sleep(Duration::from_millis(30));
        TokenCodec::decode_token(&CharCodec, id)
    }
}

async fn spawn(codec: Arc<dyn TokenCodec>, eos: Vec<u32>) -> Server {
    let model = qwen_infer::tiny_model(false);
    qwen_server::serve(model, codec, eos, "127.0.0.1:0".parse().unwrap())
        .await
        .expect("serve")
}

fn url(server: &Server, path: &str) -> String {
    format!("http://{}{path}", server.addr())
}

fn chat_body(extra: Value) -> Value {
    let mut body = json!({
        "model": qwen_server::MODEL_ID,
        "messages": [{"role": "user", "content": "Say hi"}],
        "temperature": 0,
        "max_tokens": 4
    });
    if let (Some(obj), Some(extra)) = (body.as_object_mut(), extra.as_object()) {
        for (key, value) in extra {
            obj.insert(key.clone(), value.clone());
        }
    }
    body
}

fn assert_openai_error(value: &Value, param: Option<&str>) {
    let error = &value["error"];
    assert!(error["message"].is_string(), "{value}");
    assert_eq!(error["type"], "invalid_request_error", "{value}");
    match param {
        Some(name) => assert_eq!(error["param"], name, "{value}"),
        None => assert!(error["param"].is_null(), "{value}"),
    }
    assert!(error["code"].is_null(), "{value}");
}

#[tokio::test]
async fn models_and_nonstreaming_usage_match_the_template() {
    let server = spawn(Arc::new(CharCodec), Vec::new()).await;
    let client = reqwest::Client::new();
    let listed: Value = client
        .get(url(&server, "/v1/models"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(listed["object"], "list");
    assert_eq!(listed["data"][0]["id"], qwen_server::MODEL_ID);
    assert_eq!(listed["data"][0]["object"], "model");

    let messages = vec![ChatTurn {
        role: "user".into(),
        content: "Say hi".into(),
    }];
    let prompt = apply_qwen_chat_template(&messages).unwrap();
    let expected_prompt = prompt.chars().count() as u64;

    let response = client
        .post(url(&server, "/v1/chat/completions"))
        .json(&chat_body(json!({"stream": false, "max_tokens": 4})))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], qwen_server::MODEL_ID);
    assert_eq!(body["choices"][0]["message"]["role"], "assistant");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content.chars().count(), 4, "{content}");
    assert_eq!(body["usage"]["prompt_tokens"], expected_prompt);
    assert_eq!(body["usage"]["completion_tokens"], 4);
    assert_eq!(body["usage"]["total_tokens"], expected_prompt + 4, "{body}");
    eprintln!("usage prompt {expected_prompt} completion 4 content {content:?}");
    server.shutdown().await;
}

#[tokio::test]
async fn stream_ends_with_done_and_chunk_objects() {
    let server = spawn(Arc::new(CharCodec), Vec::new()).await;
    let client = reqwest::Client::new();
    let response = client
        .post(url(&server, "/v1/chat/completions"))
        .json(&chat_body(json!({"stream": true, "max_tokens": 3})))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    let content_type = response
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        content_type.starts_with("text/event-stream"),
        "{content_type}"
    );
    let raw = response.text().await.unwrap();
    assert!(raw.contains("chat.completion.chunk"), "{raw}");
    let data: Vec<&str> = raw
        .lines()
        .filter(|line| line.starts_with("data:"))
        .collect();
    assert_eq!(data.last().copied(), Some("data: [DONE]"), "{data:?}");
    let mut contents = String::new();
    let mut saw_role = false;
    let mut saw_length = false;
    for line in &data {
        let payload = line.trim_start_matches("data:").trim();
        if payload == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(payload).unwrap();
        assert_eq!(chunk["object"], "chat.completion.chunk");
        let delta = &chunk["choices"][0]["delta"];
        if delta["role"] == "assistant" {
            saw_role = true;
        }
        if let Some(text) = delta["content"].as_str() {
            contents.push_str(text);
        }
        if chunk["choices"][0]["finish_reason"] == "length" {
            saw_length = true;
        }
        assert!(
            chunk.get("usage").is_none(),
            "usage is omitted unless include_usage is set: {chunk}"
        );
    }
    assert!(saw_role);
    assert!(saw_length);
    assert_eq!(contents.chars().count(), 3, "{contents}");
    server.shutdown().await;
}

#[tokio::test]
async fn include_usage_nulls_every_chunk_then_sends_an_empty_choices_chunk() {
    let server = spawn(Arc::new(CharCodec), Vec::new()).await;
    let client = reqwest::Client::new();
    let prompt = apply_qwen_chat_template(&[ChatTurn {
        role: "user".into(),
        content: "Say hi".into(),
    }])
    .unwrap();
    let expected_prompt = prompt.chars().count() as u64;

    let off = sse_values(
        &client,
        &server,
        &chat_body(json!({"stream": true, "max_tokens": 4})),
    )
    .await;
    assert_eq!(off.last().and_then(|v| v.as_str()), Some("[DONE]"));
    for value in &off[..off.len() - 1] {
        assert!(value.get("usage").is_none(), "{value}");
        assert!(!value["choices"].as_array().unwrap().is_empty(), "{value}");
    }

    let on = sse_values(
        &client,
        &server,
        &chat_body(json!({
            "stream": true,
            "max_tokens": 4,
            "stream_options": {"include_usage": true}
        })),
    )
    .await;
    assert_eq!(on.last().and_then(|v| v.as_str()), Some("[DONE]"));
    let json_chunks: Vec<&Value> = on[..on.len() - 1].iter().collect();
    let (usage_chunk, normals) = json_chunks.split_last().unwrap();
    assert!(
        usage_chunk["choices"].as_array().unwrap().is_empty(),
        "{usage_chunk}"
    );
    assert_eq!(usage_chunk["usage"]["prompt_tokens"], expected_prompt);
    assert_eq!(usage_chunk["usage"]["completion_tokens"], 4);
    assert_eq!(usage_chunk["usage"]["total_tokens"], expected_prompt + 4);
    assert!(!normals.is_empty());
    for value in normals {
        assert!(value["usage"].is_null(), "{value}");
        assert!(!value["choices"].as_array().unwrap().is_empty(), "{value}");
    }
    server.shutdown().await;
}

async fn sse_values(client: &reqwest::Client, server: &Server, body: &Value) -> Vec<Value> {
    let response = client
        .post(url(server, "/v1/chat/completions"))
        .json(body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200, "{body}");
    let raw = response.text().await.unwrap();
    raw.lines()
        .filter(|line| line.starts_with("data:"))
        .map(|line| {
            let payload = line.trim_start_matches("data:").trim();
            if payload == "[DONE]" {
                Value::String("[DONE]".to_string())
            } else {
                serde_json::from_str(payload).unwrap()
            }
        })
        .collect()
}

#[tokio::test]
async fn shutdown_cancels_a_stream_the_client_never_reads() {
    let server = spawn(Arc::new(CharCodec), Vec::new()).await;
    let addr = server.addr();
    let body = chat_body(json!({"stream": true, "max_tokens": 1500})).to_string();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}",
        body.len()
    );
    let tcp = std::net::TcpStream::connect(addr).unwrap();
    let sock = socket2::Socket::from(tcp);
    // The kernel raises this, but it stays far below 1500 SSE events.
    sock.set_recv_buffer_size(1024).unwrap();
    let tcp: std::net::TcpStream = sock.into();
    tcp.set_nonblocking(true).unwrap();
    let mut stream = tokio::net::TcpStream::from_std(tcp).unwrap();
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut buf = [0u8; 256];
    let _ = tokio::time::timeout(Duration::from_secs(2), stream.read(&mut buf)).await;
    // Stop reading. The engine fills the token channel and would block forever
    // on `blocking_send`. Shutdown has to notice the flag and return.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let started = std::time::Instant::now();
    let stats = tokio::time::timeout(Duration::from_secs(2), server.shutdown())
        .await
        .expect("shutdown waited on the unread stream");
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_secs(2),
        "shutdown took {elapsed:?}"
    );
    assert!(stats.last_finished, "{stats:?}");
    assert!(
        stats.last_completion_tokens < 1500,
        "generation ran to max_tokens instead of stopping: {stats:?}"
    );
    eprintln!(
        "shutdown of an unread stream took {elapsed:?} after {} completion tokens (max 1500, cancelled {})",
        stats.last_completion_tokens, stats.last_cancelled
    );
    drop(stream);
}

#[tokio::test]
async fn eos_counts_as_a_completion_token_and_is_not_decoded() {
    // The tiny net's greedy id is whatever decode emits. EOS is every id, so
    // the first sample stops the loop. That id is not decoded.
    let eos: Vec<u32> = (0..32).collect();
    let server = spawn(Arc::new(CharCodec), eos).await;
    let client = reqwest::Client::new();
    let prompt = apply_qwen_chat_template(&[ChatTurn {
        role: "user".into(),
        content: "Say hi".into(),
    }])
    .unwrap();
    let response = client
        .post(url(&server, "/v1/chat/completions"))
        .json(&chat_body(json!({"stream": false, "max_tokens": 8})))
        .send()
        .await
        .unwrap();
    let body: Value = response.json().await.unwrap();
    assert_eq!(body["choices"][0]["finish_reason"], "stop", "{body}");
    assert_eq!(body["choices"][0]["message"]["content"], "");
    assert_eq!(
        body["usage"]["prompt_tokens"],
        prompt.chars().count() as u64
    );
    assert_eq!(body["usage"]["completion_tokens"], 1);
    assert_eq!(
        body["usage"]["total_tokens"],
        body["usage"]["prompt_tokens"].as_u64().unwrap() + 1
    );
    server.shutdown().await;
}

#[tokio::test]
async fn disconnect_cancels_generation() {
    let server = spawn(Arc::new(SlowCodec), Vec::new()).await;
    let client = reqwest::Client::new();
    let mut response = client
        .post(url(&server, "/v1/chat/completions"))
        .json(&chat_body(json!({"stream": true, "max_tokens": 200})))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    // The role event is sent before any token. Keep reading until a content
    // delta arrives so the engine has actually sampled, then drop the socket.
    let mut buf = Vec::new();
    loop {
        let Some(chunk) = response.chunk().await.unwrap() else {
            break;
        };
        buf.extend_from_slice(&chunk);
        if buf.windows(9).any(|window| window == b"\"content\"") {
            break;
        }
        if buf.len() > 64 * 1024 {
            break;
        }
    }
    assert!(
        buf.windows(9).any(|window| window == b"\"content\""),
        "no content delta before drop: {}",
        String::from_utf8_lossy(&buf)
    );
    drop(response);

    let mut stats = server.stats();
    for _ in 0..100 {
        if stats.last_finished && !stats.active {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        stats = server.stats();
    }
    assert!(stats.last_finished, "{stats:?}");
    assert!(!stats.active, "{stats:?}");
    assert!(stats.last_cancelled, "{stats:?}");
    assert_eq!(stats.last_max_tokens, 200);
    assert!(
        stats.last_completion_tokens >= 1,
        "engine never sampled: {stats:?}"
    );
    assert!(
        stats.last_completion_tokens < 40,
        "engine kept generating after the client dropped: {stats:?}"
    );
    eprintln!(
        "disconnect cancelled after {} completion tokens (max {})",
        stats.last_completion_tokens, stats.last_max_tokens
    );
    server.shutdown().await;
}

#[tokio::test]
async fn bad_requests_are_400_with_an_openai_error() {
    let server = spawn(Arc::new(CharCodec), Vec::new()).await;
    let client = reqwest::Client::new();
    let post = url(&server, "/v1/chat/completions");

    let cases: Vec<(&str, Option<&str>, reqwest::RequestBuilder)> = vec![
        (
            "malformed",
            None,
            client
                .post(&post)
                .header("content-type", "application/json")
                .body("{"),
        ),
        (
            "missing messages",
            None,
            client
                .post(&post)
                .json(&json!({"model": qwen_server::MODEL_ID})),
        ),
        (
            "empty messages",
            Some("messages"),
            client.post(&post).json(&json!({
                "model": qwen_server::MODEL_ID,
                "messages": []
            })),
        ),
        (
            "missing content",
            Some("messages"),
            client.post(&post).json(&json!({
                "model": qwen_server::MODEL_ID,
                "messages": [{"role": "user"}]
            })),
        ),
        (
            "bad role",
            Some("messages"),
            client.post(&post).json(&json!({
                "model": qwen_server::MODEL_ID,
                "messages": [{"role": "tool", "content": "x"}]
            })),
        ),
        (
            "temperature",
            Some("temperature"),
            client
                .post(&post)
                .json(&chat_body(json!({"temperature": -0.4}))),
        ),
        (
            "temperature high",
            Some("temperature"),
            client
                .post(&post)
                .json(&chat_body(json!({"temperature": 2.5}))),
        ),
        (
            "top_p",
            Some("top_p"),
            client.post(&post).json(&chat_body(json!({"top_p": 1.2}))),
        ),
        (
            "top_p negative",
            Some("top_p"),
            client.post(&post).json(&chat_body(json!({"top_p": -0.1}))),
        ),
        (
            "unknown model",
            Some("model"),
            client
                .post(&post)
                .json(&chat_body(json!({"model": "gpt-4"}))),
        ),
        (
            "empty stop",
            Some("stop"),
            client.post(&post).json(&chat_body(json!({"stop": ""}))),
        ),
        (
            "n",
            Some("n"),
            client.post(&post).json(&chat_body(json!({"n": 2}))),
        ),
    ];

    for (name, param, request) in cases {
        let response = request.send().await.unwrap();
        assert_eq!(response.status(), 400, "{name}");
        let body: Value = response.json().await.unwrap();
        assert_openai_error(&body, param);
    }
    server.shutdown().await;
}

#[tokio::test]
async fn openai_sdk_streams_and_reads_usage() {
    let server = spawn(Arc::new(CharCodec), Vec::new()).await;
    let base = format!("http://{}/v1", server.addr());
    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("python3")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/openai_smoke.py"
            ))
            .arg(base)
            .arg("4")
            .output()
            .expect("python3")
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "stdout {}\nstderr {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.starts_with("OK "), "{stdout}");
    eprintln!("sdk: {}", stdout.trim());
    server.shutdown().await;
}

#[tokio::test]
#[ignore = "downloads Qwen2.5-0.5B-Instruct and serves it to the OpenAI Python SDK"]
async fn openai_sdk_against_qwen_weights() {
    let files = qwen_infer::download_qwen().expect("hf-hub");
    let tokenizer_path = files.tokenizer.to_str().expect("utf-8 path");
    let tokenizer = tokenizer::RankTokenizer::load_qwen(tokenizer_path).expect("tokenizer");
    let prompt = apply_qwen_chat_template(&[ChatTurn {
        role: "user".into(),
        content: "The capital of France is".into(),
    }])
    .unwrap();
    let ids = tokenizer.encode(&prompt).unwrap();
    assert_eq!(ids[0], 151644, "<|im_start|> is one special id");
    assert!(ids.contains(&151645), "im_end missing from {ids:?}");
    assert_eq!(tokenizer.decode(&ids).unwrap(), prompt);

    let eos = qwen_infer::load_eos_token_ids(&files.generation_config).unwrap();
    let model = qwen_infer::load_model(&files.config, &files.weights).expect("weights");
    let server = qwen_server::serve(
        model,
        Arc::new(tokenizer),
        eos,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap();
    let base = format!("http://{}/v1", server.addr());
    let output = tokio::task::spawn_blocking(move || {
        std::process::Command::new("python3")
            .arg(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/openai_smoke.py"
            ))
            .arg(base)
            .arg("8")
            .output()
            .expect("python3")
    })
    .await
    .unwrap();
    assert!(
        output.status.success(),
        "stdout {}\nstderr {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.starts_with("OK "), "{stdout}");
    eprintln!("real-weight sdk: {stdout}");
    server.shutdown().await;
}
