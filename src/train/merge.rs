use std::collections::HashMap;

pub fn top_pair(counts: &HashMap<(u32, u32), usize>) -> Option<((u32, u32), usize)> {
    if counts.is_empty(){
        return None
    }
    let mut max_count = 0;
    let mut max_pair = (0,0);
    for (key, value) in counts {
        if (*value > max_count) || ((*value == max_count) && max_pair > *key) {
            max_count = *value;
            max_pair = *key;
        } 
    }
    Some((max_pair, max_count))
}

pub fn merge(seq: &[u32], pair: (u32, u32), new_id: u32) -> Vec<u32> {
    let mut i = 0;
    let mut v: Vec<u32> = Vec::new(); 
    while i < seq.len() {
        if i < seq.len() -1 && pair == (seq[i], seq[i+1]) {
            v.push(new_id);
            i += 2;
        } else {
            v.push(seq[i]);
            i += 1;
        }
    }
    v
}