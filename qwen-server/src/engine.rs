//! One OS thread owns the model and every live [`Sequence`].
//!
//! Handlers never call `forward`. They reserve a slot and enqueue a
//! [`GenRequest`]. The engine drains that queue, admits from a FIFO waiting
//! deque while the running batch has room, and runs one iteration: prefill
//! each new sequence on its own, then one batched decode of the sequences
//! that already have a cache. Tokens go out on the per-request
//! `tokio::sync::mpsc` channel. `try_send` is the only send. A dropped
//! receiver, or a send that fails, removes that sequence before the next
//! iteration. Nothing wraps the batch in `Arc<Mutex>`.
//!
//! The thread is not a tokio worker. `blocking_send` would panic there, and
//! it would also wait forever if the client stopped reading.
//!
//! Shutdown sets an [`AtomicBool`] that this loop checks between iterations
//! and while a token send is retrying a full channel. It does not abort a
//! forward that has already entered candle. A batched matmul runs to
//! completion; the flag is observed on the next iteration boundary.

use crate::TokenCodec;
use crate::error::ApiError;
use qwen_infer::{Control, GenerateParams, KvCache, SampleCursor, SampleNext, StopReason};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// What the HTTP side last saw the engine do. Tests read this after a
/// disconnect. `last_*` is the most recently retired sequence, not a history.
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub active: bool,
    pub last_finished: bool,
    pub last_cancelled: bool,
    pub last_completion_tokens: u64,
    pub last_max_tokens: u64,
    pub requests_finished: u64,
    /// Times a per-request token send saw [`mpsc::error::TrySendError::Full`].
    /// The shutdown test reads this so a fast client is not mistaken for the
    /// channel-full backpressure path.
    pub token_channel_full: u64,
    /// Largest running batch this process has admitted.
    pub max_running: u64,
}

pub struct GenRequest {
    pub prompt_ids: Vec<u32>,
    pub params: GenerateParams,
    /// When false, deltas are not sent. The handler is waiting for [`GenUpdate::Done`].
    /// `is_closed` is still checked, so dropping that waiter cancels the run.
    pub stream: bool,
    pub tx: mpsc::Sender<GenUpdate>,
}

pub enum GenUpdate {
    Delta(String),
    Done(Done),
    Failed(String),
}

pub struct Done {
    pub text: String,
    pub reason: StopReason,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Sampled ids, excluding EOS. The HTTP body does not serialize these.
    /// The batch test compares them to a solo [`qwen_infer::generate`].
    #[allow(dead_code)]
    pub tokens: Vec<u32>,
}

/// How many sequences may be running, and how many may sit in the FIFO
/// behind them. `inflight` counts both, plus a slot a handler has reserved
/// and not yet had the engine take. Over `max_batch_size + max_waiting` the
/// handler gets HTTP 429 before the channel is touched.
pub struct Ingress {
    tx: SyncSender<GenRequest>,
    inflight: Arc<AtomicUsize>,
    max_inflight: usize,
    max_batch_size: usize,
    max_waiting: usize,
}

impl Ingress {
    pub(crate) fn new(
        tx: SyncSender<GenRequest>,
        inflight: Arc<AtomicUsize>,
        max_batch_size: usize,
        max_waiting: usize,
    ) -> Self {
        Self {
            tx,
            inflight,
            max_inflight: max_batch_size.saturating_add(max_waiting),
            max_batch_size,
            max_waiting,
        }
    }
}

pub fn try_enqueue(ingress: &Ingress, req: GenRequest) -> Result<(), ApiError> {
    let prev = ingress.inflight.fetch_add(1, Ordering::AcqRel);
    if prev >= ingress.max_inflight {
        ingress.inflight.fetch_sub(1, Ordering::AcqRel);
        return Err(ApiError::too_many(format!(
            "too many requests: max_batch_size is {}, max_waiting is {}",
            ingress.max_batch_size, ingress.max_waiting
        )));
    }
    match ingress.tx.try_send(req) {
        Ok(()) => Ok(()),
        Err(err) => {
            ingress.inflight.fetch_sub(1, Ordering::AcqRel);
            Err(map_send_error(err))
        }
    }
}

/// `Full` is the bounded `sync_channel` (the handler does not block a worker).
/// `Disconnected` means the engine thread has exited.
fn map_send_error(err: TrySendError<GenRequest>) -> ApiError {
    match err {
        TrySendError::Full(_) => ApiError::unavailable("engine queue is full"),
        TrySendError::Disconnected(_) => ApiError::server("engine is not running"),
    }
}

/// One generation the engine owns. The cache, the sampled ids, and the reply
/// channel stay on this thread. Handlers only hold the other end of `tx`.
struct Sequence {
    id: u64,
    /// Sampled ids, excluding EOS. Kept in step with [`SampleCursor::tokens`].
    tokens: Vec<u32>,
    cache: KvCache,
    tx: mpsc::Sender<GenUpdate>,
    params: GenerateParams,
    prompt: Vec<u32>,
    prompt_tokens: u32,
    stream: bool,
    phase: Phase,
    cursor: SampleCursor,
    outcome: Option<Outcome>,
    prefill: Duration,
    /// Sum of this sequence's share of each batched decode it joined.
    /// A step of wall time `t` with `n` sequences adds `t/n` here, so the
    /// shares across the batch add back up to the step.
    decode: Duration,
}

enum Phase {
    /// Cache is empty. The next iteration prefills [`Sequence::prompt`].
    Prefill,
    /// Next iteration forwards this one token at `cache.seq_len()`.
    Decode(u32),
}

enum Outcome {
    Finished {
        reason: StopReason,
        completion: usize,
    },
    Failed,
}

pub fn engine_thread(
    rx: Receiver<GenRequest>,
    model: qwen_infer::QwenModel,
    codec: Arc<dyn TokenCodec>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<Mutex<EngineStats>>,
    inflight: Arc<AtomicUsize>,
    max_batch_size: usize,
) {
    let mut waiting: VecDeque<Sequence> = VecDeque::new();
    let mut running: Vec<Sequence> = Vec::new();
    let mut next_id = 1u64;
    let mut ingress_closed = false;
    let layers = model.num_layers();

    loop {
        if shutdown.load(Ordering::SeqCst) || ingress_closed {
            break;
        }
        if running.is_empty() && waiting.is_empty() {
            match rx.recv_timeout(Duration::from_millis(100)) {
                Ok(req) => {
                    if let Some(seq) =
                        accept(req, &mut next_id, layers, &inflight, &shutdown, &stats)
                    {
                        waiting.push_back(seq);
                    }
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        loop {
            match rx.try_recv() {
                Ok(req) => {
                    if let Some(seq) =
                        accept(req, &mut next_id, layers, &inflight, &shutdown, &stats)
                    {
                        waiting.push_back(seq);
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    ingress_closed = true;
                    break;
                }
            }
        }

        // FIFO. A sequence waits only while the running batch is full.
        while running.len() < max_batch_size {
            let Some(seq) = waiting.pop_front() else {
                break;
            };
            running.push(seq);
        }
        {
            let mut guard = stats.lock().expect("engine stats");
            guard.active = !running.is_empty();
            guard.max_running = guard.max_running.max(running.len() as u64);
        }
        if running.is_empty() {
            if ingress_closed {
                break;
            }
            continue;
        }

        // A client that dropped before this iteration must not take a slot
        // in the matmul. The send-failure path below covers a drop that
        // happens during the forward: the token send fails and the sequence
        // is removed before the next iteration.
        mark_closed(&mut running);
        evict_finished(&mut running, &inflight, &stats);
        if running.is_empty() {
            stats.lock().expect("engine stats").active = false;
            continue;
        }

        step_batch(&model, codec.as_ref(), &mut running, &shutdown, &stats);
        evict_finished(&mut running, &inflight, &stats);
        stats.lock().expect("engine stats").active = !running.is_empty();
    }

    while let Ok(req) = rx.try_recv() {
        // Reserved by the handler, never turned into a Sequence.
        inflight.fetch_sub(1, Ordering::AcqRel);
        drop(req);
    }
    for seq in waiting.drain(..) {
        abandon(seq, &inflight, &stats);
    }
    for seq in running.drain(..) {
        abandon(seq, &inflight, &stats);
    }
    stats.lock().expect("engine stats").active = false;
}

fn accept(
    req: GenRequest,
    next_id: &mut u64,
    layers: usize,
    inflight: &AtomicUsize,
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
) -> Option<Sequence> {
    let id = *next_id;
    *next_id = next_id.saturating_add(1);
    if req.prompt_ids.is_empty() {
        let _ = send_update(
            &req.tx,
            shutdown,
            stats,
            GenUpdate::Failed("prompt encoded to no tokens".to_string()),
        );
        inflight.fetch_sub(1, Ordering::AcqRel);
        return None;
    }
    let cursor = match SampleCursor::new(req.params.clone()) {
        Ok(cursor) => cursor,
        Err(err) => {
            let _ = send_update(&req.tx, shutdown, stats, GenUpdate::Failed(err.to_string()));
            inflight.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
    };
    let prompt_tokens = u32::try_from(req.prompt_ids.len()).unwrap_or(u32::MAX);
    Some(Sequence {
        id,
        tokens: Vec::new(),
        cache: KvCache::new(layers),
        tx: req.tx,
        params: req.params,
        prompt: req.prompt_ids,
        prompt_tokens,
        stream: req.stream,
        phase: Phase::Prefill,
        cursor,
        outcome: None,
        prefill: Duration::ZERO,
        decode: Duration::ZERO,
    })
}

/// Prefill every new sequence by itself, then decode the rest as one batch.
///
/// Prefill is the known inefficiency. Prompts differ in length, and this
/// loop does not pack them into one matmul. A new sequence is sampled from
/// its prompt logits in this same iteration; the token it just drew is
/// forwarded on the next iteration, with whoever else is still running.
/// Shutdown cannot cut the forwards short. The flag is read when a token is
/// sent, which is after those forwards return.
fn step_batch(
    model: &qwen_infer::QwenModel,
    codec: &dyn TokenCodec,
    running: &mut Vec<Sequence>,
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
) {
    let mut parked = Vec::new();
    let mut prefilling = Vec::new();
    let mut decoding = Vec::new();
    for seq in running.drain(..) {
        if seq.outcome.is_some() {
            parked.push(seq);
            continue;
        }
        match seq.phase {
            Phase::Prefill => prefilling.push(seq),
            Phase::Decode(_) => decoding.push(seq),
        }
    }
    for seq in &mut prefilling {
        prefill_one(model, codec, seq, shutdown, stats);
    }
    if !decoding.is_empty() {
        decode_group(model, codec, &mut decoding, shutdown, stats);
    }
    running.extend(parked);
    running.extend(decoding);
    running.extend(prefilling);
}

fn prefill_one(
    model: &qwen_infer::QwenModel,
    codec: &dyn TokenCodec,
    seq: &mut Sequence,
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
) {
    let prompt = std::mem::take(&mut seq.prompt);
    let started = Instant::now();
    let forwarded = model.forward(&prompt, 0, &mut seq.cache);
    seq.prefill = started.elapsed();
    match forwarded {
        Ok(logits) => match logits.to_vec1::<f32>() {
            Ok(row) => push_sample(seq, &row, codec, shutdown, stats),
            Err(err) => fail_sequence(seq, &err, shutdown, stats),
        },
        Err(err) => fail_sequence(seq, &err, shutdown, stats),
    }
}

fn decode_group(
    model: &qwen_infer::QwenModel,
    codec: &dyn TokenCodec,
    decoding: &mut [Sequence],
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
) {
    let ids: Vec<u32> = decoding
        .iter()
        .map(|seq| match seq.phase {
            Phase::Decode(id) => id,
            Phase::Prefill => unreachable!("decode group only holds Phase::Decode"),
        })
        .collect();
    let started = Instant::now();
    let forwarded = {
        let mut refs: Vec<&mut KvCache> = decoding.iter_mut().map(|seq| &mut seq.cache).collect();
        model.forward_decode_batch(&ids, &mut refs)
    };
    let elapsed = started.elapsed();
    let share = elapsed / u32::try_from(decoding.len()).unwrap_or(1);
    match forwarded {
        Ok(logits) => match logits.to_vec2::<f32>() {
            Ok(rows) => {
                for (seq, row) in decoding.iter_mut().zip(rows) {
                    seq.decode += share;
                    push_sample(seq, &row, codec, shutdown, stats);
                }
            }
            Err(err) => {
                for seq in decoding.iter_mut() {
                    fail_sequence(seq, &err, shutdown, stats);
                }
            }
        },
        Err(err) => {
            for seq in decoding.iter_mut() {
                fail_sequence(seq, &err, shutdown, stats);
            }
        }
    }
}

fn push_sample(
    seq: &mut Sequence,
    logits: &[f32],
    codec: &dyn TokenCodec,
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
) {
    let tx = seq.tx.clone();
    let stream = seq.stream;
    let result = seq.cursor.push_logits(
        logits,
        |id| codec.decode_token(id),
        |chunk| {
            if shutdown.load(Ordering::SeqCst) || tx.is_closed() {
                return Control::Stop;
            }
            if chunk.is_empty() || !stream {
                return Control::Continue;
            }
            if send_update(&tx, shutdown, stats, GenUpdate::Delta(chunk.to_string())) {
                Control::Continue
            } else {
                Control::Stop
            }
        },
    );
    match result {
        Ok(SampleNext::Forward(id)) => {
            seq.tokens = seq.cursor.tokens().to_vec();
            seq.phase = Phase::Decode(id);
        }
        Ok(SampleNext::Done(done)) => {
            seq.tokens.clone_from(&done.tokens);
            let mut reason = done.reason;
            let completion = done.completion_tokens;
            if reason != StopReason::Cancelled {
                let update = GenUpdate::Done(Done {
                    text: done.text,
                    reason,
                    prompt_tokens: seq.prompt_tokens,
                    completion_tokens: u32::try_from(completion).unwrap_or(u32::MAX),
                    tokens: done.tokens,
                });
                // A failed Done send is the same cancel as a failed delta:
                // the client is gone, and this sequence leaves the batch.
                if !send_update(&tx, shutdown, stats, update) {
                    reason = StopReason::Cancelled;
                }
            }
            seq.outcome = Some(Outcome::Finished { reason, completion });
        }
        Err(err) => fail_sequence(seq, &err, shutdown, stats),
    }
}

fn fail_sequence(
    seq: &mut Sequence,
    err: &dyn std::fmt::Display,
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
) {
    eprintln!("engine id {} error: {err}", seq.id);
    let _ = send_update(&seq.tx, shutdown, stats, GenUpdate::Failed(err.to_string()));
    seq.outcome = Some(Outcome::Failed);
}

fn mark_closed(running: &mut [Sequence]) {
    for seq in running.iter_mut() {
        if seq.outcome.is_none() && seq.tx.is_closed() {
            seq.outcome = Some(Outcome::Finished {
                reason: StopReason::Cancelled,
                completion: seq.tokens.len(),
            });
        }
    }
}

fn evict_finished(running: &mut Vec<Sequence>, inflight: &AtomicUsize, stats: &Mutex<EngineStats>) {
    let mut retired = Vec::new();
    let mut index = 0;
    while index < running.len() {
        if running[index].outcome.is_some() {
            retired.push(running.swap_remove(index));
        } else {
            index += 1;
        }
    }
    let still = !running.is_empty();
    for seq in retired {
        retire(seq, inflight, stats, still);
    }
}

fn abandon(mut seq: Sequence, inflight: &AtomicUsize, stats: &Mutex<EngineStats>) {
    if seq.outcome.is_none() {
        seq.outcome = Some(Outcome::Finished {
            reason: StopReason::Cancelled,
            completion: seq.tokens.len(),
        });
    }
    retire(seq, inflight, stats, false);
}

fn retire(seq: Sequence, inflight: &AtomicUsize, stats: &Mutex<EngineStats>, still_active: bool) {
    inflight.fetch_sub(1, Ordering::AcqRel);
    let (cancelled, completion, reason) = match seq.outcome {
        Some(Outcome::Finished { reason, completion }) => {
            (reason == StopReason::Cancelled, completion, Some(reason))
        }
        Some(Outcome::Failed) | None => (false, 0, None),
    };
    {
        let mut guard = stats.lock().expect("engine stats");
        guard.active = still_active;
        guard.last_finished = true;
        guard.last_cancelled = cancelled;
        guard.last_completion_tokens = completion as u64;
        guard.last_max_tokens = seq.params.max_tokens as u64;
        guard.requests_finished += 1;
    }
    if let Some(reason) = reason {
        eprintln!(
            "engine id {} {reason:?} completion_tokens {completion} prefill {:.4} s decode {:.4} s",
            seq.id,
            seq.prefill.as_secs_f64(),
            seq.decode.as_secs_f64()
        );
    }
}

/// Push one update without parking the engine until the client reads.
///
/// `false` means the caller should stop the sequence. A full channel
/// increments [`EngineStats::token_channel_full`] and sleeps, so shutdown
/// is still noticed. `blocking_send` is not used.
fn send_update(
    tx: &mpsc::Sender<GenUpdate>,
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
    mut update: GenUpdate,
) -> bool {
    loop {
        if shutdown.load(Ordering::SeqCst) || tx.is_closed() {
            return false;
        }
        match tx.try_send(update) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(value)) => {
                update = value;
                stats.lock().expect("engine stats").token_channel_full += 1;
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::sync_channel;
    use tokenizer::TokenizerError;

    struct UnitCodec;

    impl TokenCodec for UnitCodec {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            let _ = text;
            Ok(Vec::new())
        }

        fn decode_token(&self, id: u32) -> Result<Vec<u8>, TokenizerError> {
            Ok(vec![b'a' + (id % 26) as u8])
        }
    }

    fn unit_bytes(id: u32) -> Result<Vec<u8>, TokenizerError> {
        TokenCodec::decode_token(&UnitCodec, id)
    }

    fn open_request(
        prompt: Vec<u32>,
        max_tokens: usize,
        stream: bool,
    ) -> (GenRequest, mpsc::Receiver<GenUpdate>) {
        let (tx, rx) = mpsc::channel(8);
        let req = GenRequest {
            prompt_ids: prompt,
            params: GenerateParams::greedy(max_tokens),
            stream,
            tx,
        };
        (req, rx)
    }

    /// `Done` is already in the channel. The sender is still alive, so this
    /// does not wait for disconnect (the join test pauses the engine first).
    fn done_already_queued(mut rx: mpsc::Receiver<GenUpdate>) -> Done {
        let deadline = Instant::now() + Duration::from_secs(5);
        while Instant::now() < deadline {
            match rx.try_recv() {
                Ok(GenUpdate::Done(value)) => return value,
                Ok(GenUpdate::Delta(_)) => {}
                Ok(GenUpdate::Failed(message)) => panic!("engine failed: {message}"),
                Err(mpsc::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        panic!("Done was not queued, decode calls still in flight");
    }

    fn take_done(mut rx: mpsc::Receiver<GenUpdate>) -> Done {
        let mut done = None;
        while let Some(update) = rx.blocking_recv() {
            match update {
                GenUpdate::Delta(_) => {}
                GenUpdate::Done(value) => done = Some(value),
                GenUpdate::Failed(message) => panic!("engine failed: {message}"),
            }
        }
        done.expect("generation ended without Done")
    }

    #[test]
    fn full_request_queue_is_unavailable_not_disconnected() {
        let (tx, rx) = sync_channel(1);
        let ingress = Ingress::new(tx.clone(), Arc::new(AtomicUsize::new(0)), 8, 8);
        try_enqueue(&ingress, open_request(vec![1, 2], 2, false).0).unwrap();
        let err = tx.try_send(open_request(vec![3], 2, false).0).unwrap_err();
        assert!(
            matches!(err, TrySendError::Full(_)),
            "receiver is still alive, so a refused send is Full"
        );
        let api = map_send_error(err);
        assert_eq!(api.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
        // Same arm through the handler path. The reservation is rolled back.
        let err = try_enqueue(&ingress, open_request(vec![4], 1, false).0).unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            ingress.inflight.load(Ordering::SeqCst),
            1,
            "the Full enqueue must not keep a slot"
        );
        // The queued request can still be received. Disconnected would not.
        assert!(rx.recv_timeout(Duration::from_secs(1)).is_ok());
        drop(rx);
        let err = try_enqueue(&ingress, open_request(vec![5], 1, false).0).unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[test]
    fn over_batch_plus_waiting_is_429() {
        let (tx, _rx) = sync_channel(8);
        let ingress = Ingress::new(tx, Arc::new(AtomicUsize::new(0)), 1, 1);
        try_enqueue(&ingress, open_request(vec![1], 2, false).0).unwrap();
        try_enqueue(&ingress, open_request(vec![2], 2, false).0).unwrap();
        let err = try_enqueue(&ingress, open_request(vec![3], 2, false).0).unwrap_err();
        assert_eq!(err.status(), axum::http::StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(ingress.inflight.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn batched_outputs_equal_solo_outputs() {
        let prompts = [
            vec![1u32],
            vec![2, 3],
            vec![4, 5, 6],
            vec![7, 8, 1, 2],
            vec![3, 3, 3, 3, 3],
            vec![9, 1],
            vec![2, 2, 4, 6, 8, 1],
            vec![5],
        ];
        // Different caps shrink the batch between iterations.
        let limits = [5usize, 1, 4, 5, 2, 5, 3, 5];
        let solo_model = qwen_infer::tiny_model(false);
        // Queue every request before the thread starts, so the first admit
        // sees all eight. `sync_channel` capacity is max_batch + max_waiting.
        let (tx, rx) = sync_channel(16);
        let inflight = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let ingress = Ingress::new(tx, Arc::clone(&inflight), 8, 8);
        let mut receivers = Vec::new();
        for (prompt, &max_tokens) in prompts.iter().zip(&limits) {
            let (req, resp) = open_request(prompt.clone(), max_tokens, false);
            try_enqueue(&ingress, req).unwrap();
            receivers.push((prompt.clone(), max_tokens, resp));
        }
        let handle = std::thread::Builder::new()
            .name("qwen-engine".into())
            .spawn({
                let shutdown = Arc::clone(&shutdown);
                let stats = Arc::clone(&stats);
                let inflight = Arc::clone(&inflight);
                move || {
                    engine_thread(
                        rx,
                        qwen_infer::tiny_model(false),
                        Arc::new(UnitCodec),
                        shutdown,
                        stats,
                        inflight,
                        8,
                    );
                }
            })
            .expect("spawn");
        for (prompt, max_tokens, resp) in receivers {
            let done = take_done(resp);
            let alone = qwen_infer::generate(
                &solo_model,
                &prompt,
                &GenerateParams::greedy(max_tokens),
                unit_bytes,
            )
            .unwrap();
            assert_eq!(done.tokens, alone.tokens, "prompt {prompt:?}");
            assert_eq!(done.reason, alone.reason);
            assert_eq!(done.text, alone.text);
            assert_eq!(done.completion_tokens as usize, alone.completion_tokens);
        }
        let stats = {
            shutdown.store(true, Ordering::SeqCst);
            drop(ingress);
            handle.join().expect("join");
            stats.lock().expect("stats").clone()
        };
        assert_eq!(
            stats.max_running, 8,
            "the eight requests were not in one batch: {stats:?}"
        );
        assert_eq!(stats.requests_finished, 8, "{stats:?}");
    }

    /// Blocks inside `decode_token` so the test can enqueue a second request
    /// while the first is already inside an iteration, then observe that the
    /// second one finishes before the first has run to `max_tokens`.
    struct PauseCodec {
        seen: Arc<AtomicUsize>,
        entered: SyncSender<usize>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl TokenCodec for PauseCodec {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            TokenCodec::encode(&UnitCodec, text)
        }

        fn decode_token(&self, id: u32) -> Result<Vec<u8>, TokenizerError> {
            let n = self.seen.fetch_add(1, Ordering::SeqCst) + 1;
            // Call 1 is the long request's first sample. Call 3 is its second
            // sample, which this scheduler reaches only after the short
            // request has been prefilled and sampled (prefill runs first).
            if n == 1 || n == 3 {
                self.entered.send(n).expect("test is listening");
                self.release
                    .lock()
                    .expect("release")
                    .recv()
                    .expect("released");
            }
            unit_bytes(id)
        }
    }

    #[test]
    fn short_request_joins_without_waiting_for_the_long_one() {
        let (entered_tx, entered_rx) = sync_channel(4);
        let (release_tx, release_rx) = sync_channel(4);
        let seen = Arc::new(AtomicUsize::new(0));
        let codec = Arc::new(PauseCodec {
            seen: Arc::clone(&seen),
            entered: entered_tx,
            release: Mutex::new(release_rx),
        });
        let (tx, rx) = sync_channel(8);
        let inflight = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let ingress = Ingress::new(tx, Arc::clone(&inflight), 4, 4);
        let (long_req, mut long_rx) = open_request(vec![1, 2, 3, 4, 5], 6, true);
        try_enqueue(&ingress, long_req).unwrap();
        let handle = std::thread::Builder::new()
            .name("qwen-engine".into())
            .spawn({
                let shutdown = Arc::clone(&shutdown);
                let stats = Arc::clone(&stats);
                let inflight = Arc::clone(&inflight);
                move || {
                    engine_thread(
                        rx,
                        qwen_infer::tiny_model(false),
                        codec,
                        shutdown,
                        stats,
                        inflight,
                        4,
                    );
                }
            })
            .expect("spawn");

        let first = entered_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap_or_else(|_| panic!("no pause, decode calls {}", seen.load(Ordering::SeqCst)));
        assert_eq!(
            first, 1,
            "first pause should be the long request's first token"
        );
        let (short_req, short_rx) = open_request(vec![8, 7], 1, true);
        try_enqueue(&ingress, short_req).unwrap();
        release_tx.send(()).unwrap();
        let third = entered_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap_or_else(|_| {
                panic!(
                    "no second pause, decode calls {}",
                    seen.load(Ordering::SeqCst)
                )
            });
        assert_eq!(
            third, 3,
            "short request should have been sampled before the long request's second token"
        );

        let short = done_already_queued(short_rx);
        let alone = qwen_infer::generate(
            &qwen_infer::tiny_model(false),
            &[8, 7],
            &GenerateParams::greedy(1),
            unit_bytes,
        )
        .unwrap();
        assert_eq!(short.tokens, alone.tokens);
        assert_eq!(short.completion_tokens, 1);

        let mut long_done = false;
        let mut long_deltas = 0usize;
        while let Ok(update) = long_rx.try_recv() {
            match update {
                GenUpdate::Delta(_) => long_deltas += 1,
                GenUpdate::Done(_) => long_done = true,
                GenUpdate::Failed(message) => panic!("{message}"),
            }
        }
        assert!(
            !long_done,
            "long request finished before the short one joined"
        );
        assert!(
            long_deltas >= 1,
            "long request never emitted its first token"
        );
        assert!(
            stats.lock().expect("stats").max_running >= 2,
            "short request never entered the running batch"
        );

        release_tx.send(()).unwrap();
        shutdown.store(true, Ordering::SeqCst);
        drop(long_rx);
        drop(ingress);
        handle.join().expect("join");
    }
}
