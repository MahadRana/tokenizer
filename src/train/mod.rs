pub mod counter;
mod fast;
pub mod training;

pub type Merges = std::collections::HashMap<(u32, u32), u32>;
pub type Vocab = std::collections::HashMap<u32, Vec<u8>>;
