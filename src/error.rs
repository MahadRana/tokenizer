use std::fmt;

/// Failure from decoding token ids, reading a training file, or parsing a
/// `.tiktoken` vocab.
///
/// `InvalidUtf8` and `Io` wrap the standard-library errors so `?` can convert
/// them with `From`. A missing token id is not a panic: it is `UnknownToken`.
/// `Parse` and `Base64` are a bad vocab line. `Base64` keeps the decoder
/// error as [`source`](std::error::Error::source). There is no `From` for it:
/// the line number has to be attached at the `map_err` call.
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
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::UnknownToken(_) | TokenizerError::Parse { .. } => None,
            TokenizerError::InvalidUtf8(err) => Some(err),
            TokenizerError::Io(err) => Some(err),
            TokenizerError::Base64 { source, .. } => Some(source),
        }
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
}
