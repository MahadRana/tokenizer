use std::collections::HashMap;

pub fn decode(encoded_text: &[u32], decoder: &HashMap<u32, Vec<u8>>) -> Result<String, std::string::FromUtf8Error>  {
    let mut seq: Vec<u8> = Vec::new();
    for id in encoded_text {
        seq.extend_from_slice(&decoder[id]);
    }
    let text = String::from_utf8(seq)?;
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::encode::encode;
    use crate::train::training::training;
    use std::fs;

    const CORPUS: &str = "the quick brown fox jumps over the lazy dog. the dog barks, the fox runs. \
        café déjà vu, naïve façade, señor niño, über straße, crème brûlée, café déjà vu. \
        日本語のテキスト、日本語のテキスト。 Привет мир, привет мир! مرحبا مرحبا \
        🙂🙂 🚀🚀 👍🏽👍🏽 👨‍👩‍👧‍👦👨‍👩‍👧‍👦 🇨🇦🇨🇦";

    fn train_on(name: &str, contents: &str) -> (HashMap<(u32, u32), u32>, HashMap<u32, Vec<u8>>) {
        let path = std::env::temp_dir().join(format!("tokenizer_test_decode_{}", name));
        fs::write(&path, contents).expect("could not write fixture");
        let result = training(path.to_str().unwrap());
        fs::remove_file(&path).ok();
        result
    }

    fn assert_round_trips(name: &str, texts: &[&str]) {
        let (encoder, decoder) = train_on(name, CORPUS);
        assert!(!encoder.is_empty());
        for text in texts {
            let tokens = encode(text, &encoder);
            assert_eq!(decode(&tokens, &decoder).unwrap(), *text);
        }
    }

    #[test]
    fn decode_of_empty_sequence_is_empty() {
        let (_, decoder) = train_on("empty", "");
        assert_eq!(decode(&[], &decoder).unwrap(), "");
    }

    #[test]
    fn decode_of_raw_bytes_is_the_text() {
        let (_, decoder) = train_on("raw", "");
        assert_eq!(decode(&[97, 98, 99], &decoder).unwrap(), "abc");
    }

    #[test]
    fn decode_expands_merged_tokens() {
        // "abababab" learns 256 = "ab", 257 = "abab"
        let (_, decoder) = train_on("merged", "abababab");
        assert_eq!(decode(&[257, 256, 99], &decoder).unwrap(), "abababc");
    }

    #[test]
    fn decode_joins_multibyte_char_split_across_tokens() {
        // "é" is the two bytes [195, 169]
        let (_, decoder) = train_on("split", "");
        assert_eq!(decode(&[195, 169], &decoder).unwrap(), "é");
    }

    #[test]
    fn decode_of_invalid_utf8_is_an_error() {
        // half of "é", and a lone continuation byte
        let (_, decoder) = train_on("invalid", "");
        assert!(decode(&[195], &decoder).is_err());
        assert!(decode(&[97, 169], &decoder).is_err());
    }

    #[test]
    fn round_trip_compresses_the_training_text() {
        let (encoder, decoder) = train_on("corpus", CORPUS);
        let tokens = encode(CORPUS, &encoder);
        assert!(tokens.len() < CORPUS.len());
        assert_eq!(decode(&tokens, &decoder).unwrap(), CORPUS);
    }

    #[test]
    fn round_trip_english() {
        assert_round_trips("english", &[
            "the quick brown fox jumps over the lazy dog",
            "Hello, World! It's 3:45pm -- costs $9.99 (approx.)",
            "a",
            "",
        ]);
    }

    #[test]
    fn round_trip_whitespace_and_control_chars() {
        assert_round_trips("whitespace", &["  leading and trailing  ", "tabs\tand\nnewlines\r\n", "\0"]);
    }

    #[test]
    fn round_trip_accented_text() {
        assert_round_trips("accents", &[
            "café déjà vu",
            "naïve façade, señor niño",
            "Über die Straße, Größe",
            "Việt Nam, Łódź, Ålesund, São Paulo",
            // combining acute accent instead of the precomposed "é"
            "cafe\u{301}",
        ]);
    }

    #[test]
    fn round_trip_non_latin_scripts() {
        assert_round_trips("scripts", &[
            "日本語のテキスト",
            "中文字符",
            "한국어",
            "Привет мир",
            "مرحبا بالعالم",
            "שלום עולם",
            "नमस्ते",
        ]);
    }

    #[test]
    fn round_trip_emojis() {
        assert_round_trips("emojis", &[
            "🙂",
            "🚀🚀🚀",
            // skin tone modifier, ZWJ family, flag
            "👍🏽",
            "👨‍👩‍👧‍👦",
            "🇨🇦",
            "mixed 🙂 text café 日本 🚀",
        ]);
    }

    #[test]
    fn round_trip_text_unseen_in_training() {
        // none of these characters appear in the corpus
        assert_round_trips("unseen", &["XYZ 0123456789 @#%^&*", "ΑΒΓ ελληνικά", "🦀🎉"]);
    }
}
