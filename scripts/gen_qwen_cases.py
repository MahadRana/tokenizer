#!/usr/bin/env python3
"""Dump Qwen2.5 token ids from Hugging Face `tokenizers` for the Rust fixture.

The Rust test `qwen_tokenizer_matches_hf` compares `RankTokenizer::load_qwen`
against this file. Ids are not hand-written. The script refuses to run if
`tokenizers` is not the version recorded below.

Usage:
    scripts/gen_qwen_cases.py [tokenizer.json] [out.jsonl]

Default tokenizer path:
    ~/.cache/tokenizer/qwen2.5-0.5b-instruct/tokenizer.json
Default output:
    tests/fixtures/qwen_cases.jsonl
"""

import json
import sys
from pathlib import Path

EXPECTED_TOKENIZERS = "0.23.3"

import tokenizers

if tokenizers.__version__ != EXPECTED_TOKENIZERS:
    sys.exit(
        f"tokenizers {tokenizers.__version__} != {EXPECTED_TOKENIZERS}; "
        "bump EXPECTED_TOKENIZERS after checking the ids"
    )

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOK = (
    Path.home() / ".cache/tokenizer/qwen2.5-0.5b-instruct/tokenizer.json"
)
DEFAULT_OUT = ROOT / "tests/fixtures/qwen_cases.jsonl"

ADDED = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<tool_call>",
    "</tool_call>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|fim_pad|>",
    "<|repo_name|>",
    "<|file_sep|>",
]


def texts():
    out = []

    def add(text):
        out.append(text)

    add("")
    add("The capital of France is")
    add("hello")
    add("Hello")
    add(" world")
    add("hello world")
    add("Hello, world!")
    add("tokenization")
    add("a")
    add("ab")
    add(" ")
    add("  ")
    add("   ")
    add("\n")
    add("\n\n")
    add("\t")
    add("\t\t")
    add("\r\n")
    add("\r\n\r\n")
    add(" \n ")
    add("end ")
    add("  hello")
    add("'s")
    add("'t")
    add("'re")
    add("'ve")
    add("'m")
    add("'ll")
    add("'d")
    add("'S")
    add("'LL")
    add("it's")
    add("don't")
    add("they're")
    add("I've")
    add("I'm")
    add("we'll")
    add("he'd")
    add("0")
    add("123")
    add("12345")
    add("999999")
    add("3.14")
    add("1,000")
    add("café")
    add("e\u0301")
    add("café déjà")
    add("naïve")
    add("a\u0308")
    add("n\u0303")
    add("é")
    add("🙂")
    add("🦀")
    add("👨‍👩‍👧‍👦")
    add("日本語")
    add("你好")
    add("Привет")
    add("...")
    add("a" * 8)
    add("a" * 40)
    add("aaaaaaaa")
    add("The quick brown fox jumps over the lazy dog.")
    add("line1\nline2\nline3")
    add("tabs\tand\tspaces ")
    add("quote: 'hello'")
    add('quote: "hello"')
    add("price is $5")
    add("C++")
    add("100%")
    add("a—b")
    add("foo\u00a0bar")
    add("mixed 123 café\n\t🙂")
    for token in ADDED:
        add(token)
        add(f"hello {token} world")
        add(f"{token}{token}")
    add("hello <|endoftext|>")
    add("<|im_start|>user\nHello<|im_end|>\n")
    add("a<|fim_prefix|>b<|fim_suffix|>")
    # Digit boundaries, contractions next to punctuation, and a few
    # code-like strings. The regex, not the merge list, decides these cuts.
    for n in range(0, 30):
        add(str(n))
        add(f"n={n}")
    words = (
        "The capital of France is Paris, and the river is the Seine. "
        "Don't stop at 12345. We'll see café at 6.\n"
    )
    add(words)
    for n in range(1, len(words), 7):
        add(words[:n])
    # Whitespace runs and punctuation the pre-tokenizer isolates.
    for spaces in range(1, 9):
        add(" " * spaces)
        add("\n" * spaces)
        add("a" + (" " * spaces) + "b")
    for ch in ".,;:!?()[]{}<>@#%^&*-+=/\\|\"'`~":
        add(ch)
        add(f"a{ch}b")
    for word in ["Paris", "France", "Seine", "Qwen", "tokenizer", "Rust", "BPE"]:
        add(word)
        add(word.lower())
        add(f" {word}")
    return out


def main():
    tok_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOK
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    tok = Tokenizer.from_file(str(tok_path))
    seen = set()
    rows = []
    for text in texts():
        if text in seen:
            continue
        seen.add(text)
        with_specials = tok.encode(text).ids
        without = tok.encode(text, add_special_tokens=False).ids
        if with_specials != without:
            sys.exit(
                "add_special_tokens changed the ids for "
                f"{text!r}: {with_specials} vs {without}"
            )
        rows.append({"text": text, "ids": with_specials})
    if len(rows) < 200:
        sys.exit(f"only {len(rows)} cases; the fixture needs at least 200")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} cases to {out_path}")


if __name__ == "__main__":
    main()
