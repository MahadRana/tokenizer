mod counter;
mod merge;
use std::collections::HashMap;

fn training(file_path: &str) -> (Vec<((u32, u32), u32)>, HashMap<u32, Vec<u8>>) {
    let text = counter::read_text(file_path);
    let bytes = text.as_bytes();
    let mut seq: Vec<u32> = bytes.iter().map(|x| u32::from(*x)).collect();
    let mut encoder = Vec::new();
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
                encoder.push(((a,b), id));
                let mut v = decoder[&a].clone();
                v.extend_from_slice(&decoder[&b]);
                decoder.insert(id, v);
            }
        }
    }
    (encoder, decoder)
} 