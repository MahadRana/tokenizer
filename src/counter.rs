use std::fs;
use std::collections::HashMap;

fn read_text(file_path: &str) -> String {
    let content = fs::read_to_string(file_path).expect("Can't find file");
    content
}

fn counts(text: &[u8]) -> HashMap<(u8, u8), usize> {
    let mut map = HashMap::new();
    for i in 0..text.len()-1 {
        *map.entry((text[i], text[i+1])).or_insert(0) += 1;
        
    }
    map
}

pub fn create_counter(file_path: &str) -> HashMap<(u8, u8), usize> {
    let text = read_text(file_path);
    let byte_text = text.as_bytes();
    counts(byte_text)
}