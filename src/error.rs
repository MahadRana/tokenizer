use std::fmt;

/// Failure from decoding token ids or reading a training file.
///
/// `InvalidUtf8` and `Io` wrap the standard-library errors so `?` can convert
/// them with `From`. A missing token id is not a panic: it is `UnknownToken`.
#[derive(Debug)]
pub enum TokenizerError {
    UnknownToken(u32),
    InvalidUtf8(std::string::FromUtf8Error),
    Io(std::io::Error),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::UnknownToken(id) => write!(f, "unknown token id {id}"),
            TokenizerError::InvalidUtf8(err) => write!(f, "invalid utf-8: {err}"),
            TokenizerError::Io(err) => write!(f, "io error: {err}"),
        }
    }
}

impl std::error::Error for TokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TokenizerError::UnknownToken(_) => None,
            TokenizerError::InvalidUtf8(err) => Some(err),
            TokenizerError::Io(err) => Some(err),
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
}
