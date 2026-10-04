use std::fs;
use std::collections::HashMap;

pub fn read_text(file_path: &str) -> String {
    let content = fs::read_to_string(file_path).expect("Can't find file");
    content
}

pub fn counts(text: &[u32]) -> HashMap<(u32, u32), usize> {
    let mut map = HashMap::new();
    for w in text.windows(2) {
        *map.entry((w[0], w[1])).or_insert(0) += 1;
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_file(name: &str, contents: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("tokenizer_test_{}", name));
        fs::write(&path, contents).expect("could not write fixture");
        path
    }

    #[test]
    fn read_text_returns_file_contents() {
        let path = temp_file("read_text", "hello world");
        let text = read_text(path.to_str().unwrap());
        fs::remove_file(&path).ok();
        assert_eq!(text, "hello world");
    }

    #[test]
    fn read_text_handles_empty_file() {
        let path = temp_file("read_text_empty", "");
        let text = read_text(path.to_str().unwrap());
        fs::remove_file(&path).ok();
        assert_eq!(text, "");
    }

    #[test]
    #[should_panic(expected = "Can't find file")]
    fn read_text_panics_on_missing_file() {
        read_text("./definitely/not/a/real/path.txt");
    }

    #[test]
    fn counts_counts_adjacent_pairs() {
        let counts = counts(&[1, 2, 3]);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts[&(1, 2)], 1);
        assert_eq!(counts[&(2, 3)], 1);
    }

    #[test]
    fn counts_accumulates_repeated_pairs() {
        // "abab" -> (a,b) twice, (b,a) once
        let counts = counts(&[1, 2, 1, 2]);
        assert_eq!(counts[&(1, 2)], 2);
        assert_eq!(counts[&(2, 1)], 1);
    }

    #[test]
    fn counts_counts_overlapping_runs() {
        // [7,7,7] has two (7,7) windows
        let counts = counts(&[7, 7, 7]);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[&(7, 7)], 2);
    }

    #[test]
    fn counts_of_single_element_is_empty() {
        assert!(counts(&[42]).is_empty());
    }

    #[test]
    fn counts_of_empty_slice_is_empty() {
        assert!(counts(&[]).is_empty());
    }
}
