## 11/02/2026
A4: `RankTokenizer::encode` splits specials out first, then runs the cl100k regex (`fancy-regex` 0.20, compiled once onto the struct) and `encode_piece` on each match. The pattern is the verbatim `pat_str` from tiktoken `openai_public.py` at `3591ff175d6a80efbe4fcc7f0e219ddd4b8c52f1`, which is also tiktoken 0.14.0's `_pat_str`. `encode_with_special_tokens` errors when a registered special is not in the allowed set. `encode_piece("")` is `[]`, and the doc says so. Fixtures (293 cases) and Pride and Prejudice (772,386 bytes, 177,179 tokens) match tiktoken. Criterion on the book: 10.003 MiB/s mean. What broke: the first book compare disagreed because Python `read_text` turned `\r\n` into `\n` while Rust kept the bytes; the generator now opens with `newline=""`. A toy vocab with a gap in the ranks was rejected by the parser before the digit-boundary assertion could run.

## 11/01/2026
A3: `RankTokenizer::load_cl100k` reads `base64(bytes) rank` lines (100,256 of them) into `HashMap<Vec<u8>, u32>` plus a `Vec<Vec<u8>>` indexed by rank. The five cl100k specials sit in their own map and match only as a whole piece. Encode walks a small `Vec` of part boundaries and merges the adjacent concatenation with the lowest rank. `Tokenizer` is unchanged: cl100k has no `(u32, u32)` merge list, so this is a second type rather than a trait that would rewrite A2 encode. Bad lines are `Parse` or `Base64`, not panics. The vocab stays in `~/.cache` (sha256 checked). Piece ids are from tiktoken 0.14.0 via `scripts/gen_cl100k_pieces.py`. Release round-trip of every vocab token was 53 ms. What broke: `piece[start..end]` is a `[u8]` place, so the lookup wants `&piece[start..end]`, and an extra `&` on `HashMap::get` is E0277 (`Borrow<&[u8]>`). Edition 2024 also rejected `find(|(_, &rank)| ...)` as an explicit deref in an implicitly borrowing pattern.

## 11/01/2026
A2: `Tokenizer` lives in the library and `main` only calls it. `decode` / `decode_bytes` return `TokenizerError` (`UnknownToken`, `InvalidUtf8`, `Io`) instead of panicking on a missing id or on a missing training file. `?` works because `From` is implemented for the wrapped std errors. Training no longer recounts the whole sequence: a linked list of indexes plus a count map, and a max-heap with lazy deletion. Tie-break is still the lowest pair, via the heap key `(count, Reverse(pair))`. The old recounting loop is `train_naive` under `#[cfg(test)]`. It matches the fast trainer on generated text and on the book (1000 merges, 238,459 tokens). Release timing is in `BENCHMARKS.md`: 149.1 ms ± 5.6 ms against a same-machine A1 remeasure of 4.173 s ± 0.048 s, about 28×. What broke: `top_pair` and `counts` became unused in the non-test build once nothing called them, so they are `#[cfg(test)]` with the oracle; `head` looked unused in release because only the debug list-walk reads it; clippy wanted `let_and_return` gone in `read_text`, a collapsed `if let`, and `is_multiple_of` in the new test text.

## 11/01/2026
A1: `train` borrows the corpus as `&str` and takes `num_merges`, instead of reading a path and always doing 1000 merges. `Merges` and `Vocab` are type aliases for the two hash maps. The loop is unchanged: count pairs, take `top_pair`, merge, record, and break when every remaining pair occurs once. `main` still does the file read, asks for 1000 merges, and prints when training stops early. Tests call `train` directly, so they no longer need temp files. The missing-file panic stays on `read_text`. `book.txt` (Pride and Prejudice) is gitignored. Release baseline is in `BENCHMARKS.md`: 4.206 s ± 0.036 s, 3.239 bytes/token.

## 10/03/2026
Got the training loop written: count pairs, merge top pair, then save pair -> ID rule in Encoder list and update Decoder hashmap with new ID, (ID, \[first's bytes, seconds' bytes\])

Obstacles: 
- Can't return a borrow of something the func owns, it got freed
- Can't index strings cause UTF-8 can be more than 1 byte, use as_bytes insead
- .iter() hands me references, use * to get vals
- map[&key] crashes if the key's missing, .get() gives an Option instead

## 10/05/2026
Wrote encode. Until a pair is not found in whole string, we look through the code, find the minimum ID pair using the encoder HashMap, replace all the values with that, then reloop. return that Vec\<u32\>

Wrote decoder. For the tokens, just replace them with the values from the decoder, then convert it to a String using String::from_utf8() converted

Obstacles: 
- contains_key and get both take references not the values themselves
- when doing a for each loop, the id is already a reference
- Result Type
    - Ok(T) -> good
    - Err(E) -> error
    - Multiple ways to handle
        - unwrap() or expect("msg")to panic on err
            - Panic = crash program, print error, exit
            - Best for quick experiments
        - unwrap_or(default)
            - dont panic, just return default on err
        - ? (BEST)
            - if value is good, then give value, else return early with Err
            - requires Result\<CorrectType, ErrorType\> return type
                - ex from code Result\<String, std::string::FromUtf8Error\>
        
