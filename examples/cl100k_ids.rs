//! Write cl100k token ids for one UTF-8 file.
//!
//! `scripts/cross_check_tiktoken.py` compares this output to Python tiktoken.
//! Arguments: `<vocab-path> <input-path>`. Ids are little-endian `u32` on
//! stdout. `RankTokenizer::encode` allows every registered special, which is
//! tiktoken `encode(text, allowed_special="all")`.

use std::io::Write;

fn main() {
    let mut args = std::env::args();
    let _argv0 = args.next();
    let vocab = args.next().unwrap_or_else(|| {
        eprintln!("usage: cl100k_ids <vocab-path> <input-path>");
        std::process::exit(2);
    });
    let input = args.next().unwrap_or_else(|| {
        eprintln!("usage: cl100k_ids <vocab-path> <input-path>");
        std::process::exit(2);
    });

    let text = std::fs::read_to_string(&input).unwrap_or_else(|err| {
        eprintln!("read {input}: {err}");
        std::process::exit(1);
    });
    let tok = tokenizer::RankTokenizer::load_cl100k(&vocab).unwrap_or_else(|err| {
        eprintln!("load {vocab}: {err}");
        std::process::exit(1);
    });
    let ids = tok.encode(&text).unwrap_or_else(|err| {
        eprintln!("encode {input}: {err}");
        std::process::exit(1);
    });

    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    for id in ids {
        out.write_all(&id.to_le_bytes()).unwrap_or_else(|err| {
            eprintln!("write ids: {err}");
            std::process::exit(1);
        });
    }
    out.flush().unwrap_or_else(|err| {
        eprintln!("flush ids: {err}");
        std::process::exit(1);
    });
}
