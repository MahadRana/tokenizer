//! OpenAI-compatible HTTP API in front of one Qwen engine thread.
//!
//! The process listens with axum. Each request encodes the chat template on
//! a tokio worker (that is a tokenizer call, not a forward) and sends the
//! token ids to the engine. The engine thread is the only place
//! [`qwen_infer::QwenModel::forward`] runs.

mod chat;
mod engine;
mod error;
mod http;

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use tokio::net::TcpListener;
use tokio::task::JoinHandle as TaskHandle;

pub use chat::{ChatTurn, TemplateError, apply_qwen_chat_template};
pub use engine::EngineStats;

use engine::engine_thread;
use tokenizer::TokenizerError;

/// Id returned by `GET /v1/models` and required on `model`.
pub const MODEL_ID: &str = "Qwen/Qwen2.5-0.5B-Instruct";

/// Encode the chat string and decode one generated id.
///
/// The handler and the engine share this. Both calls take `&self`.
/// [`tokenizer::RankTokenizer`] is `Send + Sync` (`fancy_regex::Regex` is),
/// so an `Arc<dyn TokenCodec>` can sit on the worker and on the engine thread.
pub trait TokenCodec: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    fn decode_token(&self, id: u32) -> Result<Vec<u8>, TokenizerError>;
}

impl TokenCodec for tokenizer::RankTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        tokenizer::RankTokenizer::encode(self, text)
    }

    fn decode_token(&self, id: u32) -> Result<Vec<u8>, TokenizerError> {
        self.decode_bytes(&[id])
    }
}

pub struct AppState {
    pub(crate) requests: std::sync::mpsc::SyncSender<engine::GenRequest>,
    pub(crate) codec: Arc<dyn TokenCodec>,
    pub(crate) eos: Vec<u32>,
    pub(crate) max_position: usize,
    pub(crate) ids: AtomicU64,
    pub(crate) model_id: String,
}

/// A running server. Drop sets the shutdown flag and detaches the engine
/// join so a panicked test does not block the runtime thread inside `join`
/// (the engine would still be inside `recv_timeout` until the flag is seen).
pub struct Server {
    addr: SocketAddr,
    shutdown: Arc<AtomicBool>,
    cancel: Option<tokio::sync::oneshot::Sender<()>>,
    task: Option<TaskHandle<()>>,
    engine: Option<JoinHandle<()>>,
    stats: Arc<Mutex<EngineStats>>,
}

impl Server {
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    pub fn stats(&self) -> EngineStats {
        self.stats.lock().expect("engine stats").clone()
    }

    /// Stop accepting, abort the accept loop, and wait for the engine thread.
    ///
    /// `abort` has to be polled by the runtime before the request sender is
    /// dropped. This method awaits that, then joins the engine from
    /// `spawn_blocking` so the worker that is waiting is not the worker that
    /// has to run the abort.
    pub async fn shutdown(mut self) {
        self.signal_stop();
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
        if let Some(handle) = self.engine.take() {
            let _ = tokio::task::spawn_blocking(move || handle.join()).await;
        }
    }

    fn signal_stop(&mut self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::SeqCst);
        if let Some(tx) = self.cancel.take() {
            let _ = tx.send(());
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.signal_stop();
        if let Some(task) = self.task.take() {
            task.abort();
        }
        if let Some(handle) = self.engine.take() {
            std::thread::spawn(move || {
                let _ = handle.join();
            });
        }
    }
}

/// Bind `bind` and spawn the engine thread. `model` moves to that thread.
///
/// `eos` is copied onto every request. Qwen2.5-0.5B-Instruct uses the ids
/// from `generation_config.json`.
pub async fn serve(
    model: qwen_infer::QwenModel,
    codec: Arc<dyn TokenCodec>,
    eos: Vec<u32>,
    bind: SocketAddr,
) -> anyhow::Result<Server> {
    let max_position = model.max_position_embeddings();
    let (req_tx, req_rx) = std::sync::mpsc::sync_channel(32);
    let shutdown = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(Mutex::new(EngineStats::default()));
    let engine = std::thread::Builder::new()
        .name("qwen-engine".into())
        .spawn({
            let shutdown = Arc::clone(&shutdown);
            let stats = Arc::clone(&stats);
            let codec = Arc::clone(&codec);
            move || engine_thread(req_rx, model, codec, shutdown, stats)
        })
        .map_err(|err| anyhow::anyhow!("spawn engine thread: {err}"))?;

    let state = Arc::new(AppState {
        requests: req_tx,
        codec,
        eos,
        max_position,
        ids: AtomicU64::new(1),
        model_id: MODEL_ID.to_string(),
    });
    let app = http::router(state);
    let listener = TcpListener::bind(bind)
        .await
        .map_err(|err| anyhow::anyhow!("bind {bind}: {err}"))?;
    let addr = listener
        .local_addr()
        .map_err(|err| anyhow::anyhow!("local address: {err}"))?;
    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
    let task = tokio::spawn(async move {
        let server = axum::serve(listener, app).with_graceful_shutdown(async move {
            let _ = cancel_rx.await;
        });
        if let Err(err) = server.await {
            eprintln!("http server stopped: {err}");
        }
    });
    Ok(Server {
        addr,
        shutdown,
        cancel: Some(cancel_tx),
        task: Some(task),
        engine: Some(engine),
        stats,
    })
}
