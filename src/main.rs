mod codec;
mod merge;
mod train;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: tokenizer <training-file> <text-to-encode>");
        std::process::exit(1);
    }

    let text = train::counter::read_text(&args[1]);
    let num_merges: u32 = 1000;
    let (encoder, _decoder) = train::training::train(&text, num_merges);
    if encoder.len() < num_merges as usize {
        println!(
            "stopped early: no pair has count >= 2 (learned {} of {} merges)",
            encoder.len(),
            num_merges
        );
    }
    let tokens = codec::encode::encode(&args[2], &encoder);

    println!("merges learned: {}", encoder.len());
    println!("{} bytes -> {} tokens", args[2].len(), tokens.len());
    println!("{:?}", tokens);
}
