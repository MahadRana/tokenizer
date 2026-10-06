use super::counter;
use crate::merge;
use std::collections::HashMap;

pub fn training(file_path: &str) -> (HashMap<(u32, u32), u32>, HashMap<u32, Vec<u8>>) {
    let text = counter::read_text(file_path);
    let bytes = text.as_bytes();
    let mut seq: Vec<u32> = bytes.iter().map(|x| u32::from(*x)).collect();
    let mut encoder: HashMap<(u32, u32), u32> = HashMap::new();
    let mut decoder: HashMap<u32, Vec<u8>> = HashMap::new();
    for b in 0..256 {
        decoder.insert(b, vec![b as u8]);
    }

    for i in 0..1000 {
        let counts = counter::counts(&seq);
        let top = merge::top_pair(&counts);
        let id = 256 + i;
        match top {
            None => break, 
            Some(((a,b), count)) => {
                if count == 1 {
                    break
                }
                seq = merge::merge(&seq, (a,b), id);
                encoder.insert((a,b), id);
                let mut v = decoder[&a].clone();
                v.extend_from_slice(&decoder[&b]);
                decoder.insert(id, v);
            }
        }
    }
    (encoder, decoder)
} 

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn train_on(name: &str, contents: &str) -> (HashMap<(u32, u32), u32>, HashMap<u32, Vec<u8>>) {
        let path = std::env::temp_dir().join(format!("tokenizer_test_training_{}", name));
        fs::write(&path, contents).expect("could not write fixture");
        let result = training(path.to_str().unwrap());
        fs::remove_file(&path).ok();
        result
    }

    #[test]
    fn training_seeds_decoder_with_all_bytes() {
        let (encoder, decoder) = train_on("no_repeats", "abc");
        assert!(encoder.is_empty());
        assert_eq!(decoder.len(), 256);
        for b in 0..=255u8 {
            assert_eq!(decoder[&u32::from(b)], vec![b]);
        }
    }

    #[test]
    fn training_on_empty_file_learns_nothing() {
        let (encoder, decoder) = train_on("empty", "");
        assert!(encoder.is_empty());
        assert_eq!(decoder.len(), 256);
    }

    #[test]
    fn training_merges_repeated_pair() {
        // "aaaa" -> [256, 256]; the remaining (256,256) occurs once, so stop
        let (encoder, decoder) = train_on("single_merge", "aaaa");
        assert_eq!(encoder, HashMap::from([((97, 97), 256)]));
        assert_eq!(decoder[&256], b"aa");
        assert_eq!(decoder.len(), 257);
    }

    #[test]
    fn training_builds_merges_on_earlier_merges() {
        // "abababab" -> [256; 4] -> [257, 257]
        let (encoder, decoder) = train_on("chained", "abababab");
        assert_eq!(encoder, HashMap::from([((97, 98), 256), ((256, 256), 257)]));
        assert_eq!(decoder[&256], b"ab");
        assert_eq!(decoder[&257], b"abab");
    }

    #[test]
    fn training_breaks_ties_with_lowest_pair() {
        // (97,98) and (99,100) both occur twice; the lower pair merges first,
        // so it gets the lower ID
        let (encoder, _) = train_on("tie", "cdcdabab");
        assert_eq!(encoder[&(97, 98)], 256);
        assert_eq!(encoder[&(99, 100)], 257);
    }

    #[test]
    fn training_works_on_bytes_not_chars() {
        // "é" is the two bytes [195, 169]
        let (encoder, decoder) = train_on("multibyte", "éé");
        assert_eq!(encoder, HashMap::from([((195, 169), 256)]));
        assert_eq!(decoder[&256], "é".as_bytes());
    }

    #[test]
    #[should_panic(expected = "Can't find file")]
    fn training_panics_on_missing_file() {
        training("./definitely/not/a/real/path.txt");
    }
}