//! Qwen2.5 chat template, the no-tools branch.
//!
//! The string is the `chat_template` in `tokenizer_config.json` for
//! `Qwen/Qwen2.5-0.5B-Instruct`. Tools, tool-role messages, and assistant
//! `tool_calls` are not rendered: the HTTP layer rejects them. What remains
//! is the `else` of `{%- if tools %}` plus the per-message loop and
//! `add_generation_prompt`.
//!
//! `tests/fixtures/qwen_chat_template.jinja` is that field, copied out of the
//! file. The unit test renders it with Python jinja2 (the engine Hugging
//! Face uses for this template) and compares.

use serde::Serialize;

/// One chat turn after the HTTP body has been reduced to a role and a string.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// Why [`apply_qwen_chat_template`] refused the conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    Empty,
    UnsupportedRole(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TemplateError::Empty => write!(f, "messages must not be empty"),
            TemplateError::UnsupportedRole(role) => {
                write!(f, "message role {role:?} is not supported")
            }
        }
    }
}

impl std::error::Error for TemplateError {}

const DEFAULT_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

/// Render `messages` the way `apply_chat_template(..., add_generation_prompt=True)` does
/// when `tools` is absent.
///
/// A leading system message replaces the default system line. Any other
/// system message is rendered in the loop (`not loop.first` in the template).
/// The result always ends in `<|im_start|>assistant\n`, which is the
/// generation prompt. Content is copied as-is: the template concatenates with
/// `+`, it does not HTML-escape.
pub fn apply_qwen_chat_template(messages: &[ChatTurn]) -> Result<String, TemplateError> {
    if messages.is_empty() {
        return Err(TemplateError::Empty);
    }
    for message in messages {
        if message.role != "system" && message.role != "user" && message.role != "assistant" {
            return Err(TemplateError::UnsupportedRole(message.role.clone()));
        }
    }

    let mut out = String::new();
    if messages[0].role == "system" {
        out.push_str("<|im_start|>system\n");
        out.push_str(&messages[0].content);
        out.push_str("<|im_end|>\n");
    } else {
        out.push_str("<|im_start|>system\n");
        out.push_str(DEFAULT_SYSTEM);
        out.push_str("<|im_end|>\n");
    }

    for (index, message) in messages.iter().enumerate() {
        // The first system message was already written above. The template
        // skips it with `system and not loop.first`.
        if index == 0 && message.role == "system" {
            continue;
        }
        out.push_str("<|im_start|>");
        out.push_str(&message.role);
        out.push('\n');
        out.push_str(&message.content);
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(role: &str, content: &str) -> ChatTurn {
        ChatTurn {
            role: role.to_string(),
            content: content.to_string(),
        }
    }

    #[test]
    fn user_only_inserts_the_default_system_prompt() {
        let text = apply_qwen_chat_template(&[turn("user", "The capital of France is")]).unwrap();
        assert_eq!(
            text,
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn matches_jinja_on_the_shipped_template() {
        let template = include_str!("../tests/fixtures/qwen_chat_template.jinja");
        let cases = [
            vec![turn("user", "The capital of France is")],
            vec![
                turn("system", "You are a test."),
                turn("user", "Say hi"),
                turn("assistant", "Hello"),
                turn("user", "Say bye"),
            ],
            vec![
                turn("user", "one"),
                turn("assistant", "two"),
                turn("user", "three"),
            ],
            vec![
                turn("user", "line1\nline2 \"quote\""),
                turn("assistant", ""),
                turn("user", "你好🦀"),
            ],
            vec![turn("user", "first"), turn("system", "later system")],
            vec![turn("system", "only system")],
        ];
        for messages in cases {
            let ours = apply_qwen_chat_template(&messages).unwrap();
            let theirs = jinja_render(template, &messages);
            assert_eq!(ours, theirs, "template diverged for {messages:?}");
        }
    }

    fn jinja_render(template: &str, messages: &[ChatTurn]) -> String {
        let script = r#"
import json, sys
from jinja2 import Environment, BaseLoader
raw = sys.stdin.read()
template, payload = raw.split("\n---MESSAGES---\n", 1)
env = Environment(loader=BaseLoader())
rendered = env.from_string(template).render(
    messages=json.loads(payload),
    add_generation_prompt=True,
    tools=None,
)
sys.stdout.write(rendered)
"#;
        let mut child = std::process::Command::new("python3")
            .arg("-c")
            .arg(script)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("python3");
        {
            let stdin = child.stdin.as_mut().expect("stdin");
            use std::io::Write;
            write!(stdin, "{template}\n---MESSAGES---\n").unwrap();
            serde_json::to_writer(&mut *stdin, messages).unwrap();
        }
        let output = child.wait_with_output().expect("jinja");
        assert!(
            output.status.success(),
            "jinja failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).expect("jinja utf-8")
    }
}
