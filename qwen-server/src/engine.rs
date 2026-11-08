//! One OS thread owns the model and every live [`Sequence`].
//!
//! Handlers never call `forward`. They reserve a slot and enqueue a
//! [`GenRequest`]. The engine drains that queue, admits from a FIFO waiting
//! deque while the running batch has room, and runs one iteration: prefill
//! each new sequence on its own, then one batched decode of the sequences
//! that already have a cache. Tokens go out on the per-request
//! `tokio::sync::mpsc` channel. `try_send` is attempted once per sequence
//! per iteration. A full channel parks that sequence; it does not sleep the
//! thread. A dropped receiver, or a client that stays full past
//! [`SLOW_CLIENT_TIMEOUT`], is removed before the next forward. Nothing
//! wraps the batch in `Arc<Mutex>`.
//!
//! The thread is not a tokio worker. `blocking_send` would panic there.
//! Retrying one full channel in a loop would stall every other sequence,
//! including `Done` for a request that is ready to finish.
//!
//! Shutdown sets an [`AtomicBool`] that this loop checks between iterations.
//! It does not abort a forward that has already entered candle. A batched
//! matmul runs to completion; the flag is observed on the next iteration
//! boundary.

use crate::TokenCodec;
use crate::error::ApiError;
use qwen_infer::{Control, GenerateParams, KvCache, SampleCursor, SampleNext, StopReason};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TryRecvError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Unsent updates kept for one sequence. The HTTP channel holds one item.
/// This cap absorbs a client that is a few chunks behind. Past it the
/// sequence is cancelled rather than given a longer queue: generating into
/// an unread buffer pins a batch slot and the bytes.
const MAX_PENDING_UPDATES: usize = 8;

/// How long one sequence may keep an update the client has not taken.
///
/// The engine does not sleep on that sequence. Each iteration tries the send
/// once. This deadline is only when that try keeps failing, so a stuck
/// client loses its slot and the others keep the thread. Two seconds is
/// longer than one real-weight decode step (~0.2 s) and longer than a
/// scheduling gap, and short enough that a client which never reads does
/// not occupy `max_batch_size` for the life of the process.
const SLOW_CLIENT_TIMEOUT: Duration = Duration::from_secs(2);

/// When every running sequence is waiting on its client, block here for a
/// new request instead of spinning on `try_send`. One wait for the thread,
/// shared by every blocked sequence. A sequence that can run is never delayed
/// by another sequence's full channel.
const BLOCKED_RETRY_WAIT: Duration = Duration::from_millis(5);

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
    /// Times a per-request `try_send` saw [`mpsc::error::TrySendError::Full`].
    /// Counted once per failed attempt. The engine does not retry that send
    /// in a loop, so this is not a spin counter.
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
    /// Updates the client has not accepted yet. Empty means the sequence may
    /// run a forward. Non-empty means the last `try_send` was [`mpsc::error::TrySendError::Full`]
    /// and this sequence waits for a later iteration.
    pending: VecDeque<GenUpdate>,
    /// Set on the first failed flush of the current backlog. Cleared when
    /// `pending` drains. Compared to [`SLOW_CLIENT_TIMEOUT`].
    blocked_since: Option<Instant>,
    /// Generation has finished, but the terminal `Done` or `Failed` is still
    /// sitting in `pending`. Moved to `outcome` once the client takes it,
    /// which is when the reservation is released.
    held_outcome: Option<Outcome>,
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
        // in the matmul. `flush_all` is one `try_send` per blocked sequence.
        // It does not sleep. A sequence that is still full stays in `running`
        // and is not forwarded.
        mark_closed(&mut running);
        flush_all(&mut running, &shutdown, &stats);
        evict_finished(&mut running, &inflight, &stats);
        if running.is_empty() {
            stats.lock().expect("engine stats").active = false;
            if ingress_closed {
                break;
            }
            continue;
        }
        let runnable = running.iter().any(can_forward);
        if !runnable {
            stats.lock().expect("engine stats").active = true;
            // Nobody can run a forward. Wait for a new request, or a short
            // shared timeout, then retry every blocked send once. This is
            // the only sleep, and it is not inside one sequence.
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            if !ingress_closed {
                match rx.recv_timeout(BLOCKED_RETRY_WAIT) {
                    Ok(req) => {
                        if let Some(seq) =
                            accept(req, &mut next_id, layers, &inflight, &shutdown, &stats)
                        {
                            waiting.push_back(seq);
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {}
                    Err(RecvTimeoutError::Disconnected) => ingress_closed = true,
                }
            }
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
        let _ = try_send_once(
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
            let _ = try_send_once(&req.tx, shutdown, stats, GenUpdate::Failed(err.to_string()));
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
        pending: VecDeque::new(),
        blocked_since: None,
        held_outcome: None,
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
/// Shutdown cannot cut the forwards short. The flag is read after those
/// forwards return, when the new update is flushed.
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
        // Output-blocked sequences are not forwarded. Their unsent update is
        // retried by `flush_all` on the next iteration, once, with everyone else.
        if seq.outcome.is_some() || !seq.pending.is_empty() {
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
    let queued = seq.pending.len();
    let mut deltas = Vec::new();
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
            // The chunk is queued and flushed after `push_logits` returns.
            // Stopping here is the cap, not a full channel: a full channel
            // parks the sequence instead of cancelling it on the first miss.
            if queued + deltas.len() + 1 > MAX_PENDING_UPDATES {
                return Control::Stop;
            }
            deltas.push(GenUpdate::Delta(chunk.to_string()));
            Control::Continue
        },
    );
    seq.pending.extend(deltas);
    match result {
        Ok(SampleNext::Forward(id)) => {
            seq.tokens = seq.cursor.tokens().to_vec();
            seq.phase = Phase::Decode(id);
            flush_pending(seq, shutdown, stats);
        }
        Ok(SampleNext::Done(done)) => {
            seq.tokens.clone_from(&done.tokens);
            let completion = done.completion_tokens;
            if done.reason == StopReason::Cancelled {
                seq.pending.clear();
                seq.held_outcome = None;
                seq.outcome = Some(Outcome::Finished {
                    reason: StopReason::Cancelled,
                    completion,
                });
            } else if seq.pending.len() >= MAX_PENDING_UPDATES {
                cancel_slow(seq);
            } else {
                seq.pending.push_back(GenUpdate::Done(Done {
                    text: done.text,
                    reason: done.reason,
                    prompt_tokens: seq.prompt_tokens,
                    completion_tokens: u32::try_from(completion).unwrap_or(u32::MAX),
                    tokens: done.tokens,
                }));
                // Delivered inside `flush_pending` once `pending` is empty.
                // Until then the sequence stays running and is not forwarded.
                seq.held_outcome = Some(Outcome::Finished {
                    reason: done.reason,
                    completion,
                });
                flush_pending(seq, shutdown, stats);
            }
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
    if seq.outcome.is_some() {
        return;
    }
    // `Failed` is terminal output, same as `Done`. A full channel keeps it
    // in `pending` and `flush_all` retries it once per iteration. The
    // sequence stays reserved until that send lands, or until the client
    // drops or [`SLOW_CLIENT_TIMEOUT`] cancels it. Retiring here would drop
    // the sender, and an SSE client that still had one delta buffered would
    // read that delta and then EOF, with no error event and no `[DONE]`.
    // Oldest unsent updates are dropped only to keep the error inside the cap.
    while seq.pending.len() >= MAX_PENDING_UPDATES {
        seq.pending.pop_front();
    }
    seq.pending.push_back(GenUpdate::Failed(err.to_string()));
    seq.held_outcome = Some(Outcome::Failed);
    flush_pending(seq, shutdown, stats);
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

fn can_forward(seq: &Sequence) -> bool {
    seq.outcome.is_none() && seq.pending.is_empty()
}

fn flush_all(running: &mut [Sequence], shutdown: &AtomicBool, stats: &Mutex<EngineStats>) {
    for seq in running.iter_mut() {
        flush_pending(seq, shutdown, stats);
    }
}

/// Send queued updates until one does not fit.
///
/// A full channel is recorded and the update stays at the front of
/// `pending`. The caller returns to the scheduler. The next iteration tries
/// this sequence once more, after every other sequence has had the same
/// chance. There is no sleep on this path.
///
/// The sequence is cancelled when the backlog would pass
/// [`MAX_PENDING_UPDATES`] or the oldest unsent update is older than
/// [`SLOW_CLIENT_TIMEOUT`]. That is a deadline checked on a failed try, not
/// a wait. A client that is merely a chunk behind is parked, not dropped.
fn flush_pending(seq: &mut Sequence, shutdown: &AtomicBool, stats: &Mutex<EngineStats>) {
    if seq.outcome.is_some() {
        return;
    }
    if shutdown.load(Ordering::SeqCst) {
        cancel_slow(seq);
        return;
    }
    loop {
        let Some(update) = seq.pending.pop_front() else {
            seq.blocked_since = None;
            if let Some(outcome) = seq.held_outcome.take() {
                seq.outcome = Some(outcome);
            }
            return;
        };
        match seq.tx.try_send(update) {
            Ok(()) => {}
            Err(mpsc::error::TrySendError::Closed(_)) => {
                cancel_slow(seq);
                return;
            }
            Err(mpsc::error::TrySendError::Full(value)) => {
                seq.pending.push_front(value);
                note_full(stats);
                let since = seq.blocked_since.get_or_insert_with(Instant::now);
                if seq.pending.len() > MAX_PENDING_UPDATES || since.elapsed() >= SLOW_CLIENT_TIMEOUT
                {
                    cancel_slow(seq);
                }
                return;
            }
        }
    }
}

fn cancel_slow(seq: &mut Sequence) {
    let completion = seq
        .held_outcome
        .as_ref()
        .map(|outcome| match outcome {
            Outcome::Finished { completion, .. } => *completion,
            Outcome::Failed => seq.tokens.len(),
        })
        .unwrap_or(seq.tokens.len());
    seq.pending.clear();
    seq.held_outcome = None;
    seq.blocked_since = None;
    seq.outcome = Some(Outcome::Finished {
        reason: StopReason::Cancelled,
        completion,
    });
}

fn note_full(stats: &Mutex<EngineStats>) {
    stats.lock().expect("engine stats").token_channel_full += 1;
}

/// One attempt, for a message that is not part of a sequence backlog
/// (a request rejected before it is admitted).
/// `false` means it was not delivered. The caller does not retry.
/// A forward error is not sent here: [`fail_sequence`] queues it.
fn try_send_once(
    tx: &mpsc::Sender<GenUpdate>,
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
    update: GenUpdate,
) -> bool {
    if shutdown.load(Ordering::SeqCst) || tx.is_closed() {
        return false;
    }
    match tx.try_send(update) {
        Ok(()) => true,
        Err(mpsc::error::TrySendError::Closed(_)) => false,
        Err(mpsc::error::TrySendError::Full(_)) => {
            note_full(stats);
            false
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

    /// A receiver that is connected and never read fills its channel. The
    /// engine records `Full` and keeps running the other request. That
    /// request's `Done` arrives before this test drains or drops the stalled
    /// receiver.
    #[test]
    fn stalled_receiver_full_lets_the_other_request_finish() {
        let (tx, rx) = sync_channel(8);
        let inflight = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let ingress = Ingress::new(tx, Arc::clone(&inflight), 4, 4);

        // Capacity 1, so the second delta is `TrySendError::Full`.
        let (stall_tx, mut stall_rx) = mpsc::channel(1);
        try_enqueue(
            &ingress,
            GenRequest {
                prompt_ids: vec![1, 2, 3, 4],
                params: GenerateParams::greedy(40),
                stream: true,
                tx: stall_tx,
            },
        )
        .unwrap();
        let (ok_req, ok_rx) = open_request(vec![8, 7, 6], 3, false);
        try_enqueue(&ingress, ok_req).unwrap();

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
                        4,
                    );
                }
            })
            .expect("spawn");

        let done = take_done(ok_rx);
        let alone = qwen_infer::generate(
            &qwen_infer::tiny_model(false),
            &[8, 7, 6],
            &GenerateParams::greedy(3),
            unit_bytes,
        )
        .unwrap();
        assert_eq!(done.tokens, alone.tokens);
        assert_eq!(done.reason, alone.reason);

        let stats_now = stats.lock().expect("stats").clone();
        assert!(
            stats_now.token_channel_full > 0,
            "stalled receiver never returned Full: {stats_now:?}"
        );
        assert_eq!(
            stats_now.requests_finished, 1,
            "the stalled request must still be in the batch: {stats_now:?}"
        );
        assert!(stats_now.active, "{stats_now:?}");

        // One delta fit in the channel. Reading that one item is the check
        // that the sender is still alive. The following recv must be Empty:
        // the rest is parked on the engine, and the receiver has not been
        // dropped. This test does not read any further.
        assert!(
            matches!(stall_rx.try_recv(), Ok(GenUpdate::Delta(_))),
            "the single buffered delta was not sitting unread"
        );
        assert!(
            matches!(stall_rx.try_recv(), Err(mpsc::error::TryRecvError::Empty)),
            "stalled receiver was drained or dropped"
        );

        shutdown.store(true, Ordering::SeqCst);
        drop(stall_rx);
        drop(ingress);
        handle.join().expect("join");
    }

    /// The token channel is already full when generation fails. `Failed`
    /// stays pending, a peer request still finishes, and the reservation
    /// drops only after the client has taken the error. That update is the
    /// SSE error event; [`crate::http::failure_frame`] is the `[DONE]` that
    /// follows it.
    #[test]
    fn failed_while_full_is_delivered_and_the_peer_still_runs() {
        let model = qwen_infer::tiny_model(false);
        let fail_prompt = vec![1u32, 2, 3, 4];
        let peer_prompt = vec![9u32, 8, 7, 5];
        let fail_ids =
            qwen_infer::generate(&model, &fail_prompt, &GenerateParams::greedy(2), unit_bytes)
                .unwrap()
                .tokens;
        let peer_ids =
            qwen_infer::generate(&model, &peer_prompt, &GenerateParams::greedy(4), unit_bytes)
                .unwrap()
                .tokens;
        let poison = fail_ids[1];
        assert_ne!(fail_ids[0], poison, "first token must occupy the channel");
        assert!(
            !peer_ids.contains(&poison),
            "peer ids {peer_ids:?} collide with poison {poison}"
        );

        let (tx, rx) = sync_channel(8);
        let inflight = Arc::new(AtomicUsize::new(0));
        let shutdown = Arc::new(AtomicBool::new(false));
        let stats = Arc::new(Mutex::new(EngineStats::default()));
        let ingress = Ingress::new(tx, Arc::clone(&inflight), 4, 4);

        let (fail_tx, mut fail_rx) = mpsc::channel(1);
        try_enqueue(
            &ingress,
            GenRequest {
                prompt_ids: fail_prompt,
                params: GenerateParams::greedy(8),
                stream: true,
                tx: fail_tx,
            },
        )
        .unwrap();
        let (peer_req, peer_rx) = open_request(peer_prompt.clone(), 4, false);
        try_enqueue(&ingress, peer_req).unwrap();
        assert_eq!(inflight.load(Ordering::SeqCst), 2);

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
                        Arc::new(PoisonCodec { poison }),
                        shutdown,
                        stats,
                        inflight,
                        4,
                    );
                }
            })
            .expect("spawn");

        let peer = take_done(peer_rx);
        assert_eq!(peer.tokens, peer_ids);
        let stats_now = stats.lock().expect("stats").clone();
        assert!(
            stats_now.token_channel_full > 0,
            "Failed was not attempted against a full channel: {stats_now:?}"
        );
        assert_eq!(
            stats_now.requests_finished, 1,
            "the failed request must still be reserved: {stats_now:?}"
        );
        assert_eq!(inflight.load(Ordering::SeqCst), 1);
        assert!(stats_now.active, "{stats_now:?}");

        // The delta that filled the channel is still unread. Taking it is
        // what lets the parked `Failed` send on the next fair retry.
        assert!(
            matches!(fail_rx.try_recv(), Ok(GenUpdate::Delta(_))),
            "expected the unread delta that made the channel Full"
        );
        let failed = wait_update(&mut fail_rx);
        let GenUpdate::Failed(message) = failed else {
            panic!("expected the error update, got something else");
        };
        assert!(message.contains(&poison.to_string()), "{message}");
        let (error_event, done) = crate::http::failure_frame(&message);
        let parsed: serde_json::Value = serde_json::from_str(&error_event).unwrap();
        assert_eq!(parsed["error"]["type"], "server_error");
        assert_eq!(parsed["error"]["message"], message);
        assert_eq!(done, "[DONE]");

        let deadline = Instant::now() + Duration::from_secs(2);
        while inflight.load(Ordering::SeqCst) != 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            inflight.load(Ordering::SeqCst),
            0,
            "reservation held after the error was delivered"
        );
        let stats_done = stats.lock().expect("stats").clone();
        assert_eq!(stats_done.requests_finished, 2, "{stats_done:?}");
        assert!(!stats_done.active, "{stats_done:?}");

        shutdown.store(true, Ordering::SeqCst);
        drop(fail_rx);
        drop(ingress);
        handle.join().expect("join");
    }

    struct PoisonCodec {
        poison: u32,
    }

    impl TokenCodec for PoisonCodec {
        fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
            TokenCodec::encode(&UnitCodec, text)
        }

        fn decode_token(&self, id: u32) -> Result<Vec<u8>, TokenizerError> {
            if id == self.poison {
                Err(TokenizerError::UnknownToken(id))
            } else {
                unit_bytes(id)
            }
        }
    }

    fn wait_update(rx: &mut mpsc::Receiver<GenUpdate>) -> GenUpdate {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            match rx.try_recv() {
                Ok(update) => return update,
                Err(mpsc::error::TryRecvError::Empty) => {
                    std::thread::sleep(Duration::from_millis(5))
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    panic!("sender dropped before Failed was delivered")
                }
            }
        }
        panic!("timed out waiting for the error update");
    }
}
