mod codec;
mod merge;
mod train;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: tokenizer <training-file> <text-to-encode>");
        std::process::exit(1);
    }

    let (encoder, _decoder) = train::training::training(&args[1]);
    let tokens = codec::encode::encode(&args[2], &encoder);

    println!("merges learned: {}", encoder.len());
    println!("{} bytes -> {} tokens", args[2].len(), tokens.len());
    println!("{:?}", tokens);
}
