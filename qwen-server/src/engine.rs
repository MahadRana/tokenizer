//! One OS thread owns the model.
//!
//! Handlers never call `forward`. They enqueue a [`GenRequest`] and wait on
//! the per-request `tokio::sync::mpsc` channel. The engine thread is not a
//! tokio worker. `blocking_send` would panic there, and it would also wait
//! forever if the client stopped reading, so this thread uses `try_send`.
//!
//! A dropped receiver is the cancel signal. The chunk callback returns
//! [`Control::Stop`], and [`generate_with`] does not run the next forward.
//! Token sends use `try_send` and sleep, not `blocking_send`: a client that
//! never reads fills the channel, and a blocking send would pin this thread
//! across shutdown. The request queue is a bounded `std::sync::mpsc` so the
//! engine can `recv_timeout` and notice shutdown without entering the runtime.
//! `try_send` from a handler does not block a worker if the queue is full.

use crate::TokenCodec;
use qwen_infer::{Control, GenerateParams, StopReason, generate_with};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, TrySendError};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc;

/// What the HTTP side last saw the engine do. Tests read this after a
/// disconnect. It describes the most recent request, not a history.
#[derive(Debug, Clone, Default)]
pub struct EngineStats {
    pub active: bool,
    pub last_finished: bool,
    pub last_cancelled: bool,
    pub last_completion_tokens: u64,
    pub last_max_tokens: u64,
    pub requests_finished: u64,
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
}

pub fn try_enqueue(
    tx: &std::sync::mpsc::SyncSender<GenRequest>,
    req: GenRequest,
) -> Result<(), crate::error::ApiError> {
    match tx.try_send(req) {
        Ok(()) => Ok(()),
        Err(TrySendError::Full(_)) => {
            Err(crate::error::ApiError::unavailable("engine queue is full"))
        }
        Err(TrySendError::Disconnected(_)) => {
            Err(crate::error::ApiError::server("engine is not running"))
        }
    }
}

pub fn engine_thread(
    rx: Receiver<GenRequest>,
    model: qwen_infer::QwenModel,
    codec: Arc<dyn TokenCodec>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<Mutex<EngineStats>>,
) {
    loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(req) => serve_one(&model, codec.as_ref(), &shutdown, &stats, req),
            Err(RecvTimeoutError::Timeout) => continue,
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }
}

/// Push one update without parking the engine until the client reads.
///
/// `blocking_send` on a full channel waits until the SSE task takes the item.
/// A client that never reads stops that task at the socket buffer, so the
/// wait never ends and shutdown's `join` never returns. `try_send` plus a
/// short sleep lets this thread notice [`AtomicBool`] between attempts.
/// `false` means the caller should stop the request.
fn send_update(tx: &mpsc::Sender<GenUpdate>, shutdown: &AtomicBool, mut update: GenUpdate) -> bool {
    loop {
        if shutdown.load(Ordering::SeqCst) || tx.is_closed() {
            return false;
        }
        match tx.try_send(update) {
            Ok(()) => return true,
            Err(mpsc::error::TrySendError::Closed(_)) => return false,
            Err(mpsc::error::TrySendError::Full(value)) => {
                update = value;
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
}

fn serve_one(
    model: &qwen_infer::QwenModel,
    codec: &dyn TokenCodec,
    shutdown: &AtomicBool,
    stats: &Mutex<EngineStats>,
    req: GenRequest,
) {
    let max_tokens = req.params.max_tokens;
    {
        let mut guard = stats.lock().expect("engine stats");
        guard.active = true;
        guard.last_finished = false;
        guard.last_cancelled = false;
        guard.last_completion_tokens = 0;
        guard.last_max_tokens = max_tokens as u64;
    }
    if req.tx.is_closed() {
        finish(stats, true, 0, max_tokens);
        return;
    }

    let prompt_tokens = u32::try_from(req.prompt_ids.len()).unwrap_or(u32::MAX);
    let tx = &req.tx;
    let stream = req.stream;
    let result = generate_with(
        model,
        &req.prompt_ids,
        &req.params,
        |id| codec.decode_token(id),
        |chunk| {
            // Checked on empty pieces too, so a shutdown during a withheld
            // UTF-8 tail still stops before the next forward.
            if shutdown.load(Ordering::SeqCst) || tx.is_closed() {
                return Control::Stop;
            }
            if chunk.is_empty() || !stream {
                return Control::Continue;
            }
            if send_update(tx, shutdown, GenUpdate::Delta(chunk.to_string())) {
                Control::Continue
            } else {
                Control::Stop
            }
        },
    );

    match result {
        Ok(generated) => {
            let cancelled = generated.reason == StopReason::Cancelled;
            let completion = generated.completion_tokens;
            eprintln!(
                "engine {:?} completion_tokens {completion} prefill {:.4} s decode {:.4} s",
                generated.reason,
                generated.prefill.as_secs_f64(),
                generated.decode.as_secs_f64()
            );
            if !cancelled {
                let done = GenUpdate::Done(Done {
                    text: generated.text,
                    reason: generated.reason,
                    prompt_tokens,
                    completion_tokens: u32::try_from(completion).unwrap_or(u32::MAX),
                });
                if !send_update(tx, shutdown, done) {
                    finish(stats, true, completion, max_tokens);
                    return;
                }
            }
            finish(stats, cancelled, completion, max_tokens);
        }
        Err(err) => {
            eprintln!("engine error: {err:#}");
            let _ = send_update(tx, shutdown, GenUpdate::Failed(err.to_string()));
            finish(stats, false, 0, max_tokens);
        }
    }
}

fn finish(stats: &Mutex<EngineStats>, cancelled: bool, completion: usize, max_tokens: usize) {
    let mut guard = stats.lock().expect("engine stats");
    guard.active = false;
    guard.last_finished = true;
    guard.last_cancelled = cancelled;
    guard.last_completion_tokens = completion as u64;
    guard.last_max_tokens = max_tokens as u64;
    guard.requests_finished += 1;
}
