//! Streaming detokenizer for byte-level BPE.
//!
//! One token can be a prefix of a UTF-8 character (`Tokenizer::decode_bytes`
//! returns those bytes and does not invent a replacement). This buffer holds
//! an incomplete tail and emits a `&str` chunk only when the codepoint is
//! finished. A truncated sequence is `Utf8Error::error_len() == None`. That
//! tail stays buffered. A complete illegal sequence is the only place a
//! U+FFFD is produced, and only for those bytes, not for a character that
//! might still arrive.

/// Bytes of tokens seen so far that are not yet valid UTF-8.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Bytes held because they may still be a prefix of a codepoint.
    pub fn pending(&self) -> &[u8] {
        &self.pending
    }

    /// Push the raw bytes of one token. The returned string is the newly
    /// completed text and does not include [`Self::pending`].
    pub fn push(&mut self, token_bytes: &[u8]) -> String {
        self.pending.extend_from_slice(token_bytes);
        let mut out = String::new();
        loop {
            if self.pending.is_empty() {
                break;
            }
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    out.push_str(text);
                    self.pending.clear();
                    break;
                }
                Err(err) => {
                    let valid = err.valid_up_to();
                    if valid > 0 {
                        // `valid_up_to` is a character boundary.
                        if let Ok(text) = std::str::from_utf8(&self.pending[..valid]) {
                            out.push_str(text);
                        }
                        self.pending.drain(..valid);
                        continue;
                    }
                    match err.error_len() {
                        None => break,
                        Some(len) => {
                            out.push('\u{FFFD}');
                            let drop_n = len.min(self.pending.len());
                            self.pending.drain(..drop_n);
                        }
                    }
                }
            }
        }
        out
    }

    /// End of the stream. An incomplete tail becomes U+FFFD here, not in
    /// [`Self::push`].
    pub fn finish(&mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        rest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_bytes(decoder: &mut StreamDecoder, bytes: &[u8]) -> String {
        // Same call a generation loop makes: one id, then `decode_bytes`.
        let tok = tokenizer::Tokenizer::train("", 0);
        let mut out = String::new();
        for &byte in bytes {
            let piece = tok.decode_bytes(&[u32::from(byte)]).unwrap();
            out.push_str(&decoder.push(&piece));
        }
        out
    }

    #[test]
    fn stream_decoder_buffers_partial_utf8() {
        // 2-byte é is c3 a9. 3-byte 中 is e4 b8 ad. 4-byte 🦀 is f0 9f a6 80.
        let text = "aé中🦀b";
        let mut decoder = StreamDecoder::new();
        let mut acc = String::new();
        let tok = tokenizer::Tokenizer::train("", 0);
        let mut saw_held_tail = false;
        for &byte in text.as_bytes() {
            let piece = tok.decode_bytes(&[u32::from(byte)]).unwrap();
            let chunk = decoder.push(&piece);
            assert!(
                !chunk.contains('\u{FFFD}'),
                "replacement emitted mid-stream on {chunk:?} after byte {byte:#x}, pending {:02x?}",
                decoder.pending()
            );
            if !decoder.pending().is_empty() {
                saw_held_tail = true;
                // The held tail is a proper prefix of some UTF-8 character,
                // so it is not itself valid UTF-8.
                assert!(std::str::from_utf8(decoder.pending()).is_err());
            }
            acc.push_str(&chunk);
        }
        assert!(saw_held_tail, "every character arrived in one byte");
        assert!(decoder.pending().is_empty());
        assert_eq!(acc, text);
        assert_eq!(decoder.finish(), "");

        // Each width on its own, split across separate byte tokens.
        let cases = ["é", "中", "🦀", "a中b🦀"];
        for case in cases {
            let mut decoder = StreamDecoder::new();
            let got = push_bytes(&mut decoder, case.as_bytes());
            assert_eq!(got, case, "pending {:?}", decoder.pending());
            assert!(decoder.pending().is_empty());
        }

        // A 4-byte lead with nothing after it stays buffered and does not
        // become U+FFFD until finish().
        let mut decoder = StreamDecoder::new();
        assert_eq!(decoder.push(&[0xF0]), "");
        assert_eq!(decoder.pending(), &[0xF0]);
        assert_eq!(decoder.push(&[0x9F, 0xA6]), "");
        assert_eq!(decoder.finish(), "\u{FFFD}");
        assert!(decoder.pending().is_empty());
    }
}
