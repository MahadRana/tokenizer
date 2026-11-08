fn main() -> Result<(), tokenizer::TokenizerError> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: tokenizer <training-file> <text-to-encode>");
        std::process::exit(1);
    }

    let text = tokenizer::read_text(&args[1])?;
    let num_merges: u32 = 1000;
    let model = tokenizer::Tokenizer::train(&text, num_merges);
    let learned = model.merge_count();
    if learned < num_merges as usize {
        println!(
            "stopped early: no pair has count >= 2 (learned {learned} of {num_merges} merges)"
        );
    }
    let tokens = model.encode(&args[2]);

    println!("merges learned: {learned}");
    println!("{} bytes -> {} tokens", args[2].len(), tokens.len());
    println!("{tokens:?}");
    Ok(())
}
