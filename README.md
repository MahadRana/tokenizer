# tokenizer

Byte-level BPE.

`Tokenizer::train` learns a merge list from text. `RankTokenizer::load_cl100k` loads OpenAI's cl100k vocab (`bytes -> rank`) and encodes one pre-split piece. The vocab file is not in git.

## Tests

`cargo test` does not use the network and does not need the vocab file or `book.txt`.

The cl100k file is cached at `~/.cache/tokenizer/cl100k_base.tiktoken`. Download it with:

```bash
scripts/fetch-cl100k.sh
```

`cargo test --release -- --ignored` downloads that file with `curl` when it is missing, checks the sha256 published with tiktoken, and also runs `fast_training_matches_naive_on_book`, which needs `book.txt` in the repo root (gitignored).

The pairs in `encode_piece_matches_known_ids` come from `scripts/gen_cl100k_pieces.py` (tiktoken 0.14.0, `Encoding._encode_single_piece`). That is piece encoding, with no regex split. Full-text parity is a later milestone.
