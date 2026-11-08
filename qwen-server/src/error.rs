//! OpenAI error body: `{"error": {"message", "type", "param", "code"}}`.
//!
//! axum's [`Json`] extractor answers a bad body with 422. The chat API this
//! server is copying uses 400, so the handler takes
//! `Result<Json<T>, JsonRejection>` and maps every rejection here.

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Serialize;

#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
    kind: &'static str,
    param: Option<String>,
    code: Option<String>,
}

impl ApiError {
    pub fn invalid(message: impl Into<String>, param: Option<&str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            kind: "invalid_request_error",
            param: param.map(str::to_string),
            code: None,
        }
    }

    pub fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            kind: "server_error",
            param: None,
            code: None,
        }
    }

    pub fn unavailable(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: message.into(),
            kind: "server_error",
            param: None,
            code: None,
        }
    }

    /// Batch or waiting-queue capacity is exhausted.
    ///
    /// OpenAI's error guide (fetched 2026-11-08) uses HTTP 429 with
    /// `type: rate_limit_error` for "slow down", and the error object still
    /// carries `message`, `type`, `param`, and `code`. `rate_limit_exceeded`
    /// is the code on that object when the limit is a request cap.
    /// <https://platform.openai.com/docs/guides/error-codes>
    pub fn too_many(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::TOO_MANY_REQUESTS,
            message: message.into(),
            kind: "rate_limit_error",
            param: None,
            code: Some("rate_limit_exceeded".to_string()),
        }
    }

    /// Status the HTTP layer will send. Unit tests read this. The binary
    /// itself only goes through [`IntoResponse`].
    #[cfg(test)]
    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn from_rejection(err: JsonRejection) -> Self {
        // `body_text` is the message axum would have written. The status on
        // the rejection is 422 or 415; the chat API still wants 400.
        Self::invalid(err.body_text(), None)
    }
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    error: ErrorObject<'a>,
}

#[derive(Serialize)]
struct ErrorObject<'a> {
    message: &'a str,
    #[serde(rename = "type")]
    kind: &'a str,
    param: Option<&'a str>,
    code: Option<&'a str>,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = ErrorBody {
            error: ErrorObject {
                message: &self.message,
                kind: self.kind,
                param: self.param.as_deref(),
                code: self.code.as_deref(),
            },
        };
        (self.status, Json(body)).into_response()
    }
}
