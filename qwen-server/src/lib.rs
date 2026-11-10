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

use engine::{Ingress, engine_thread};

/// Admission limits for the iteration scheduler.
///
/// `max_batch_size` is how many sequences run a forward together.
/// `max_waiting` is how many more may sit in the FIFO. A request past both
/// is HTTP 429. The defaults match the old queue depth (32 waiting) and a
/// batch of 8, which is the size the throughput check uses.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub max_batch_size: usize,
    pub max_waiting: usize,
    /// Physical KV blocks shared by every sequence. Each block holds
    /// [`qwen_infer::BLOCK_SIZE`] tokens. The default is 256 blocks, which is
    /// 4096 tokens of cache.
    pub num_blocks: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            max_waiting: 32,
            num_blocks: 256,
        }
    }
}
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
    pub(crate) requests: Ingress,
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
    /// The flag is set before the accept loop is aborted, so an engine blocked
    /// on a full token channel observes it and leaves. The returned stats are
    /// whatever the engine wrote before that thread exited.
    pub async fn shutdown(mut self) -> EngineStats {
        self.signal_stop();
        if let Some(task) = self.task.take() {
            task.abort();
            let _ = task.await;
        }
        if let Some(handle) = self.engine.take() {
            let _ = tokio::task::spawn_blocking(move || handle.join()).await;
        }
        self.stats()
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
/// from `generation_config.json`. Admission uses [`ServerConfig::default`].
pub async fn serve(
    model: qwen_infer::QwenModel,
    codec: Arc<dyn TokenCodec>,
    eos: Vec<u32>,
    bind: SocketAddr,
) -> anyhow::Result<Server> {
    serve_with(model, codec, eos, bind, ServerConfig::default()).await
}

/// [`serve`] with an explicit batch and waiting-queue limit.
pub async fn serve_with(
    model: qwen_infer::QwenModel,
    codec: Arc<dyn TokenCodec>,
    eos: Vec<u32>,
    bind: SocketAddr,
    config: ServerConfig,
) -> anyhow::Result<Server> {
    if config.max_batch_size == 0 {
        anyhow::bail!("max_batch_size must be at least 1");
    }
    if config.num_blocks == 0 {
        anyhow::bail!("num_blocks must be at least 1");
    }
    let max_position = model.max_position_embeddings();
    let pool = model
        .kv_pool(config.num_blocks)
        .map_err(|err| anyhow::anyhow!("paged kv pool: {err}"))?;
    let max_inflight = config.max_batch_size.saturating_add(config.max_waiting);
    // The channel is at least as deep as the admission cap, so a reserved
    // slot can always be queued. `TrySendError::Full` is then the defensive
    // path (the counter and the channel disagree), not the 429 path.
    let (req_tx, req_rx) = std::sync::mpsc::sync_channel(max_inflight.max(1));
    let shutdown = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(Mutex::new(EngineStats::default()));
    let inflight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let engine = std::thread::Builder::new()
        .name("qwen-engine".into())
        .spawn({
            let shutdown = Arc::clone(&shutdown);
            let stats = Arc::clone(&stats);
            let codec = Arc::clone(&codec);
            let inflight = Arc::clone(&inflight);
            let max_batch_size = config.max_batch_size;
            move || {
                engine_thread(
                    req_rx,
                    model,
                    codec,
                    shutdown,
                    stats,
                    inflight,
                    max_batch_size,
                    pool,
                )
            }
        })
        .map_err(|err| anyhow::anyhow!("spawn engine thread: {err}"))?;
    let requests = Ingress::new(req_tx, inflight, config.max_batch_size, config.max_waiting);

    let state = Arc::new(AppState {
        requests,
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
