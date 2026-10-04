use std::fs;
use std::collections::HashMap;

pub fn read_text(file_path: &str) -> String {
    let content = fs::read_to_string(file_path).expect("Can't find file");
    content
}

pub fn counts(text: &[u32]) -> HashMap<(u32, u32), usize> {
    let mut map = HashMap::new();
    for i in 0..text.len()-1 {
        *map.entry((text[i], text[i+1])).or_insert(0) += 1;
        
    }
    map
}

