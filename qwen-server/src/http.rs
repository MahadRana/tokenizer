//! `POST /v1/chat/completions` and `GET /v1/models`.
//!
//! `Json` is the last extractor. axum's `Handler` bound is not implemented
//! when a body extractor sits in front of `State`. A parse failure is a
//! `JsonRejection` (422 from axum) and is turned into a 400 before the
//! engine hears about it.
//!
//! The SSE body is a [`ReceiverStream`] plus a short queue of events. `map`
//! can only emit one item per token. A finished stream emits the finish
//! chunk, then (when `stream_options.include_usage` is set) a chunk whose
//! `choices` are empty and whose `usage` is the token counts, then
//! `data: [DONE]`. The docs for that shape are the Chat Completions reference,
//! `stream_options.include_usage`, fetched 2026-11-07.

use crate::chat::{ChatTurn, apply_qwen_chat_template};
use crate::engine::{self, Done, GenRequest, GenUpdate};
use crate::error::ApiError;
use crate::{AppState, MODEL_ID};
use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use qwen_infer::{GenerateParams, StopReason};
use serde::ser::Serializer;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::Stream;
use tokio_stream::wrappers::ReceiverStream;

const DEFAULT_MAX_TOKENS: u32 = 128;
const MAX_TOKENS_CAP: u32 = 8192;
/// 2024-09-19, the Qwen2.5-0.5B-Instruct release day, as a unix second.
const MODEL_CREATED: u64 = 1_726_704_000;

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .with_state(state)
}

#[derive(Debug, Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Incoming>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    stop: Option<StopField>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    n: Option<u32>,
    #[serde(default)]
    stream_options: Option<StreamOptions>,
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StopField {
    One(String),
    Many(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct Incoming {
    role: String,
    #[serde(default)]
    content: Option<Content>,
    #[serde(default)]
    tool_calls: Option<Vec<serde_json::Value>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Deserialize)]
struct ContentPart {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
}

struct Prepared {
    prompt_ids: Vec<u32>,
    params: GenerateParams,
    stream: bool,
    include_usage: bool,
    id: String,
    created: u64,
}

async fn list_models() -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![ModelCard {
            id: MODEL_ID,
            object: "model",
            created: MODEL_CREATED,
            owned_by: "qwen",
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    payload: Result<Json<ChatRequest>, JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(request) = payload.map_err(ApiError::from_rejection)?;
    let prepared = prepare(&state, request)?;
    let (tx, rx) = mpsc::channel(1);
    let stream = prepared.stream;
    let include_usage = prepared.include_usage;
    let id = prepared.id.clone();
    let created = prepared.created;
    engine::try_enqueue(
        &state.requests,
        GenRequest {
            prompt_ids: prepared.prompt_ids,
            params: prepared.params,
            stream,
            tx,
        },
    )?;
    if stream {
        let body = CompletionSse {
            inner: ReceiverStream::new(rx),
            pending: VecDeque::new(),
            phase: Phase::Role,
            id,
            created,
            include_usage,
        };
        Ok(Sse::new(body)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let done = wait_done(rx).await?;
        Ok(Json(completion_body(&id, created, done)).into_response())
    }
}

async fn wait_done(mut rx: mpsc::Receiver<GenUpdate>) -> Result<Done, ApiError> {
    let mut done = None;
    while let Some(update) = rx.recv().await {
        match update {
            GenUpdate::Delta(_) => {}
            GenUpdate::Done(value) => done = Some(value),
            GenUpdate::Failed(message) => return Err(ApiError::server(message)),
        }
    }
    done.ok_or_else(|| ApiError::server("generation ended without a result"))
}

fn prepare(state: &AppState, request: ChatRequest) -> Result<Prepared, ApiError> {
    if request.model != state.model_id {
        return Err(ApiError::invalid(
            format!("The model `{}` does not exist", request.model),
            Some("model"),
        ));
    }
    if request.messages.is_empty() {
        return Err(ApiError::invalid(
            "messages must not be empty",
            Some("messages"),
        ));
    }
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(ApiError::invalid("tools are not supported", Some("tools")));
    }
    if request.n.is_some_and(|n| n != 1) {
        return Err(ApiError::invalid("n must be 1", Some("n")));
    }
    let temperature = check_unit("temperature", request.temperature.unwrap_or(1.0), 0.0, 2.0)?;
    let top_p = check_unit("top_p", request.top_p.unwrap_or(1.0), 0.0, 1.0)?;
    let max_tokens = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(DEFAULT_MAX_TOKENS);
    if max_tokens > MAX_TOKENS_CAP {
        return Err(ApiError::invalid(
            format!("max_tokens must be <= {MAX_TOKENS_CAP}"),
            Some("max_tokens"),
        ));
    }
    let stop = stop_list(request.stop)?;
    let mut turns = Vec::with_capacity(request.messages.len());
    for message in request.messages {
        if message
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
        {
            return Err(ApiError::invalid(
                "tool_calls are not supported",
                Some("messages"),
            ));
        }
        let content = message_content(message.content)?;
        if message.role != "system" && message.role != "user" && message.role != "assistant" {
            return Err(ApiError::invalid(
                format!("message role {:?} is not supported", message.role),
                Some("messages"),
            ));
        }
        turns.push(ChatTurn {
            role: message.role,
            content,
        });
    }
    let prompt = apply_qwen_chat_template(&turns)
        .map_err(|err| ApiError::invalid(err.to_string(), Some("messages")))?;
    let prompt_ids = state.codec.encode(&prompt).map_err(|err| {
        ApiError::invalid(
            format!("failed to tokenize the prompt: {err}"),
            Some("messages"),
        )
    })?;
    if prompt_ids.is_empty() {
        return Err(ApiError::invalid(
            "prompt encoded to no tokens",
            Some("messages"),
        ));
    }
    let prompt_len = prompt_ids.len();
    if prompt_len > state.max_position {
        return Err(ApiError::invalid(
            format!(
                "prompt is {prompt_len} tokens and the context limit is {}",
                state.max_position
            ),
            Some("messages"),
        ));
    }
    let room = state.max_position - prompt_len;
    if max_tokens as usize > room {
        return Err(ApiError::invalid(
            format!("max_tokens leaves the context limit ({room} tokens of room)"),
            Some("max_tokens"),
        ));
    }

    let mut params = GenerateParams::greedy(max_tokens as usize);
    params.temperature = temperature;
    params.top_p = top_p;
    params.seed = request.seed.unwrap_or(0);
    params.eos = state.eos.clone();
    params.stop = stop;

    let n = state.ids.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    Ok(Prepared {
        prompt_ids,
        params,
        stream: request.stream,
        include_usage: request
            .stream_options
            .is_some_and(|options| options.include_usage),
        id: format!("chatcmpl-{n:08x}"),
        created,
    })
}

fn message_content(content: Option<Content>) -> Result<String, ApiError> {
    let Some(content) = content else {
        return Err(ApiError::invalid(
            "message content is required",
            Some("messages"),
        ));
    };
    match content {
        Content::Text(text) => Ok(text),
        Content::Parts(parts) => {
            if parts.is_empty() {
                return Err(ApiError::invalid(
                    "message content is empty",
                    Some("messages"),
                ));
            }
            let mut out = String::new();
            for part in parts {
                if part.kind.as_deref().unwrap_or("text") != "text" {
                    return Err(ApiError::invalid(
                        "only text content parts are supported",
                        Some("messages"),
                    ));
                }
                let Some(text) = part.text else {
                    return Err(ApiError::invalid(
                        "text content part is missing text",
                        Some("messages"),
                    ));
                };
                out.push_str(&text);
            }
            Ok(out)
        }
    }
}

fn stop_list(stop: Option<StopField>) -> Result<Vec<String>, ApiError> {
    let list = match stop {
        None => Vec::new(),
        Some(StopField::One(value)) => vec![value],
        Some(StopField::Many(values)) => values,
    };
    if list.len() > 4 {
        return Err(ApiError::invalid(
            "stop supports at most 4 sequences",
            Some("stop"),
        ));
    }
    if list.iter().any(|value| value.is_empty()) {
        return Err(ApiError::invalid(
            "stop sequences must be non-empty",
            Some("stop"),
        ));
    }
    Ok(list)
}

fn check_unit(name: &str, value: f32, low: f32, high: f32) -> Result<f32, ApiError> {
    if !value.is_finite() || !(low..=high).contains(&value) {
        return Err(ApiError::invalid(
            format!("{name} must be between {low} and {high}, got {value}"),
            Some(name),
        ));
    }
    Ok(value)
}

#[derive(Serialize)]
struct ModelList {
    object: &'static str,
    data: Vec<ModelCard>,
}

#[derive(Serialize)]
struct ModelCard {
    id: &'static str,
    object: &'static str,
    created: u64,
    owned_by: &'static str,
}

#[derive(Serialize)]
struct CompletionBody {
    id: String,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

#[derive(Serialize)]
struct CompletionChoice {
    index: u32,
    message: OutMessage,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct OutMessage {
    role: &'static str,
    content: String,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

fn completion_body(id: &str, created: u64, done: Done) -> CompletionBody {
    CompletionBody {
        id: id.to_string(),
        object: "chat.completion",
        created,
        model: MODEL_ID,
        choices: vec![CompletionChoice {
            index: 0,
            message: OutMessage {
                role: "assistant",
                content: done.text,
            },
            finish_reason: finish_reason(done.reason),
        }],
        usage: usage(done.prompt_tokens, done.completion_tokens),
    }
}

fn usage(prompt_tokens: u32, completion_tokens: u32) -> Usage {
    Usage {
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens.saturating_add(completion_tokens),
    }
}

fn finish_reason(reason: StopReason) -> &'static str {
    match reason {
        StopReason::MaxTokens => "length",
        StopReason::Eos | StopReason::StopString | StopReason::Cancelled => "stop",
    }
}

/// `include_usage` off omits the field. On, every ordinary chunk is JSON
/// `null` and the extra final chunk carries the object. `Option` cannot say
/// both "omit" and "null".
enum UsageField {
    Omit,
    Null,
    Present(Usage),
}

fn usage_field(include_usage: bool) -> UsageField {
    if include_usage {
        UsageField::Null
    } else {
        UsageField::Omit
    }
}

fn omit_usage(field: &UsageField) -> bool {
    matches!(field, UsageField::Omit)
}

impl Serialize for UsageField {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            UsageField::Omit | UsageField::Null => serializer.serialize_none(),
            UsageField::Present(usage) => usage.serialize(serializer),
        }
    }
}

#[derive(Serialize)]
struct Chunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: Vec<ChunkChoice<'a>>,
    #[serde(skip_serializing_if = "omit_usage")]
    usage: UsageField,
}

#[derive(Serialize)]
struct ChunkChoice<'a> {
    index: u32,
    delta: Delta<'a>,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct Delta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

enum Phase {
    Role,
    Events,
    Done,
}

struct CompletionSse {
    inner: ReceiverStream<GenUpdate>,
    pending: VecDeque<Event>,
    phase: Phase,
    id: String,
    created: u64,
    include_usage: bool,
}

impl Stream for CompletionSse {
    type Item = Result<Event, Infallible>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // `ReceiverStream` is `Unpin`. The other fields are too, so this
        // stream is `Unpin` and `get_mut` does not need a projection.
        let this = self.get_mut();
        if let Some(event) = this.pending.pop_front() {
            return Poll::Ready(Some(Ok(event)));
        }
        if matches!(this.phase, Phase::Done) {
            return Poll::Ready(None);
        }
        if matches!(this.phase, Phase::Role) {
            this.phase = Phase::Events;
            return Poll::Ready(Some(Ok(event_json(&chunk_role(
                &this.id,
                this.created,
                this.include_usage,
            )))));
        }
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                this.phase = Phase::Done;
                Poll::Ready(None)
            }
            Poll::Ready(Some(GenUpdate::Delta(text))) => Poll::Ready(Some(Ok(event_json(
                &chunk_content(&this.id, this.created, &text, this.include_usage),
            )))),
            Poll::Ready(Some(GenUpdate::Done(done))) => {
                // Ordinary chunks, including this finish chunk, carry `usage:
                // null` when include_usage is set. The counts ride a following
                // chunk with an empty `choices` array, then `[DONE]`.
                if this.include_usage {
                    this.pending.push_back(event_json(&chunk_usage(
                        &this.id,
                        this.created,
                        done.prompt_tokens,
                        done.completion_tokens,
                    )));
                }
                this.pending.push_back(Event::default().data("[DONE]"));
                this.phase = Phase::Done;
                Poll::Ready(Some(Ok(event_json(&chunk_finish(
                    &this.id,
                    this.created,
                    finish_reason(done.reason),
                    this.include_usage,
                )))))
            }
            Poll::Ready(Some(GenUpdate::Failed(message))) => {
                let (body, done) = failure_frame(&message);
                this.pending.push_back(Event::default().data(done));
                this.phase = Phase::Done;
                Poll::Ready(Some(Ok(Event::default().data(body))))
            }
        }
    }
}

fn chunk_role(id: &str, created: u64, include_usage: bool) -> Chunk<'_> {
    Chunk {
        id,
        object: "chat.completion.chunk",
        created,
        model: MODEL_ID,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: Some("assistant"),
                content: None,
            },
            finish_reason: None,
        }],
        usage: usage_field(include_usage),
    }
}

fn chunk_content<'a>(id: &'a str, created: u64, text: &'a str, include_usage: bool) -> Chunk<'a> {
    Chunk {
        id,
        object: "chat.completion.chunk",
        created,
        model: MODEL_ID,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: Some(text),
            },
            finish_reason: None,
        }],
        usage: usage_field(include_usage),
    }
}

fn chunk_finish<'a>(
    id: &'a str,
    created: u64,
    reason: &'static str,
    include_usage: bool,
) -> Chunk<'a> {
    Chunk {
        id,
        object: "chat.completion.chunk",
        created,
        model: MODEL_ID,
        choices: vec![ChunkChoice {
            index: 0,
            delta: Delta {
                role: None,
                content: None,
            },
            finish_reason: Some(reason),
        }],
        usage: usage_field(include_usage),
    }
}

fn chunk_usage(id: &str, created: u64, prompt_tokens: u32, completion_tokens: u32) -> Chunk<'_> {
    Chunk {
        id,
        object: "chat.completion.chunk",
        created,
        model: MODEL_ID,
        choices: Vec::new(),
        usage: UsageField::Present(usage(prompt_tokens, completion_tokens)),
    }
}

/// SSE payload for a generation failure, then the terminator the stream
/// queues behind it. One `GenUpdate::Failed` is this JSON event and then
/// `[DONE]`. The engine must deliver that update; dropping the sender
/// instead ends the stream on EOF.
pub(crate) fn failure_frame(message: &str) -> (String, &'static str) {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "param": null,
            "code": null
        }
    });
    (body.to_string(), "[DONE]")
}

fn event_json(value: &impl Serialize) -> Event {
    match Event::default().json_data(value) {
        Ok(event) => event,
        Err(err) => Event::default().data(format!(
            "{{\"error\":{{\"message\":\"failed to encode SSE chunk: {err}\",\"type\":\"server_error\",\"param\":null,\"code\":null}}}}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;

    #[tokio::test]
    async fn failed_update_is_an_error_event_then_done() {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tx.try_send(GenUpdate::Failed("forward exploded".into()))
            .unwrap();
        drop(tx);
        let sse = CompletionSse {
            inner: ReceiverStream::new(rx),
            pending: VecDeque::new(),
            phase: Phase::Role,
            id: "chatcmpl-test".into(),
            created: 1,
            include_usage: false,
        };
        let response = Sse::new(sse).into_response();
        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("sse body");
        let text = String::from_utf8(bytes.to_vec()).unwrap();
        let error_at = text.find("server_error").expect(&text);
        let done_at = text.find("[DONE]").expect(&text);
        assert!(error_at < done_at, "{text}");
        assert!(text.contains("forward exploded"), "{text}");
    }
}
