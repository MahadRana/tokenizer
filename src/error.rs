use std::fmt;

/// Failure from decoding token ids, reading a training file, parsing a
/// `.tiktoken` vocab, or encoding text.
///
/// `InvalidUtf8`, `Io`, and `Regex` wrap lower-level errors so `?` can convert
/// them with `From`. A missing token id is not a panic: it is `UnknownToken`.
/// `Parse` and `Base64` are a bad vocab line. `Base64` keeps the decoder
/// error as [`source`](std::error::Error::source). There is no `From` for it:
/// the line number has to be attached at the `map_err` call.
/// `DisallowedSpecial` is a special token in the text that the caller did not
/// allow. It is an error, not a byte-pair encoding of that text.
#[derive(Debug)]
pub enum TokenizerError {
    UnknownToken(u32),
    InvalidUtf8(std::string::FromUtf8Error),
    Io(std::io::Error),
    /// A vocab line is not `base64(bytes) rank`, or the ranks are not `0..n`.
    ///
    /// `line` is 1-based. Line 0 means the file as a whole (the wrong number
    /// of tokens, or a missing single-byte token).
    Parse {
        line: usize,
        message: String,
    },
    /// The base64 token on this line did not decode. `line` is 1-based.
    Base64 {
        line: usize,
        source: base64::DecodeError,
    },
    /// `text` contains this registered special token, and it was not in the
    /// allowed set passed to [`crate::RankTokenizer::encode_with_special_tokens`].
    DisallowedSpecial(String),
    /// The pre-tokenizer regex failed to compile, or `find_iter` failed
    /// while searching (fancy-regex reports a backtrack limit this way).
    Regex(fancy_regex::Error),
    /// `tokenizer.json` is not valid JSON. The line number, when serde has
    /// one, is inside the source error.
    Json(serde_json::Error),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::UnknownToken(id) => write!(f, "unknown token id {id}"),
            TokenizerError::InvalidUtf8(err) => write!(f, "invalid utf-8: {err}"),
            TokenizerError::Io(err) => write!(f, "io error: {err}"),
            TokenizerError::Parse { line, message } => {
                if *line == 0 {
                    write!(f, "parse error: {message}")
                } else {
                    write!(f, "parse error on line {line}: {message}")
                }
            }
            TokenizerError::Base64 { line, source } => {
                write!(f, "base64 error on line {line}: {source}")
            }
            TokenizerError::DisallowedSpecial(token) => {
                write!(f, "disallowed special token {token:?}")
            }
            TokenizerError::Regex(err) => write!(f, "regex error: {err}"),
            TokenizerError::Json(err) => write!(f, "json error: {err}"),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::UnknownToken(_)
            | TokenizerError::Parse { .. }
            | TokenizerError::DisallowedSpecial(_) => None,
            TokenizerError::InvalidUtf8(err) => Some(err),
            TokenizerError::Io(err) => Some(err),
            TokenizerError::Base64 { source, .. } => Some(source),
            TokenizerError::Regex(err) => Some(err),
            TokenizerError::Json(err) => Some(err),
        }
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(err: serde_json::Error) -> Self {
        TokenizerError::Json(err)
    }
}

impl From<fancy_regex::Error> for TokenizerError {
    fn from(err: fancy_regex::Error) -> Self {
        TokenizerError::Regex(err)
    }
}

impl From<std::string::FromUtf8Error> for TokenizerError {
    fn from(err: std::string::FromUtf8Error) -> Self {
        TokenizerError::InvalidUtf8(err)
    }
}

impl From<std::io::Error> for TokenizerError {
    fn from(err: std::io::Error) -> Self {
        TokenizerError::Io(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::io;

    #[test]
    fn question_mark_converts_utf8_error() {
        fn decode_owned(bytes: Vec<u8>) -> Result<String, TokenizerError> {
            // `?` calls `From::from` on the `FromUtf8Error`.
            Ok(String::from_utf8(bytes)?)
        }

        let err = decode_owned(vec![0xff]).unwrap_err();
        assert!(matches!(err, TokenizerError::InvalidUtf8(_)));
        assert!(err.source().is_some());
        let rendered = err.to_string();
        assert!(rendered.contains("utf-8"), "{rendered}");
    }

    #[test]
    fn io_error_converts_and_keeps_source() {
        let err = TokenizerError::from(io::Error::new(io::ErrorKind::NotFound, "missing"));
        assert!(matches!(err, TokenizerError::Io(_)));
        assert!(err.source().is_some());
        assert!(err.to_string().contains("missing"));
    }

    #[test]
    fn unknown_token_has_no_source() {
        let err = TokenizerError::UnknownToken(42);
        assert!(err.source().is_none());
        assert_eq!(err.to_string(), "unknown token id 42");
    }

    #[test]
    fn parse_error_names_the_line_and_has_no_source() {
        let err = TokenizerError::Parse {
            line: 3,
            message: "missing space".into(),
        };
        assert!(err.source().is_none());
        assert_eq!(err.to_string(), "parse error on line 3: missing space");

        let whole_file = TokenizerError::Parse {
            line: 0,
            message: "expected 100256 tokens, found 2".into(),
        };
        assert_eq!(
            whole_file.to_string(),
            "parse error: expected 100256 tokens, found 2"
        );
    }

    #[test]
    fn base64_error_keeps_the_decoder_as_source() {
        use base64::Engine as _;
        use base64::engine::general_purpose::STANDARD;

        let source = STANDARD.decode("!!!!").unwrap_err();
        let err = TokenizerError::Base64 { line: 4, source };
        assert!(err.source().is_some());
        let rendered = err.to_string();
        assert!(rendered.contains("line 4"), "{rendered}");
        assert!(rendered.contains("base64"), "{rendered}");
    }

    #[test]
    fn question_mark_converts_regex_error() {
        fn compile_bad() -> Result<(), TokenizerError> {
            // `?` calls `From::from` on `fancy_regex::Error`.
            let _re = fancy_regex::Regex::new("(")?;
            Ok(())
        }

        let err = compile_bad().unwrap_err();
        assert!(matches!(err, TokenizerError::Regex(_)));
        assert!(err.source().is_some());
        assert!(err.to_string().contains("regex"), "{err}");
    }

    #[test]
    fn disallowed_special_names_the_token_and_has_no_source() {
        let err = TokenizerError::DisallowedSpecial("<|endoftext|>".into());
        assert!(err.source().is_none());
        assert_eq!(
            err.to_string(),
            "disallowed special token \"<|endoftext|>\""
        );
    }
}
