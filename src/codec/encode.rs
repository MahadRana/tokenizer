use crate::merge;
use std::collections::HashMap;

pub fn encode(text: &str, encoder: &HashMap<(u32, u32), u32>) -> Vec<u32> {
    let bytes = text.as_bytes();
    let mut seq: Vec<u32> = bytes.iter().map(|x| u32::from(*x)).collect();
    if seq.is_empty(){
        return seq;
    }

    let mut found = true; 
    while found {
        found = false;
        let mut min_pair = (u32::MAX,u32::MAX);
        let mut min_id = u32::MAX;
        for i in 0..seq.len()-1 {
            let pair = (seq[i], seq[i+1]);
            if encoder.contains_key(&pair) {
                found = true;
                let id = encoder[&pair];
                if id < min_id {
                    min_id = id;
                    min_pair = pair;
                }
            }
        }
        if found {
            seq = merge::merge(&seq, min_pair, min_id);
        }
    }
    seq
}