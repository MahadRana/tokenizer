#!/usr/bin/env python3
"""Print the (piece, ids) pairs committed in the A3 tests.

Oracle: Python tiktoken, encoding ``cl100k_base``. Requires
``pip install tiktoken`` (this repo was generated with tiktoken 0.14.0).

Ordinary pieces use ``Encoding._encode_single_piece``. That is byte-pair
encode of one already-split piece: no regex, and no special-token match.
``Encoding.encode`` splits on the cl100k regex first. The script prints
both. Where they differ, the Rust test must follow ``_encode_single_piece``.
Full-text parity is milestone A4, not this one.

Specials use ``Encoding.encode(..., allowed_special={token})``.
``_encode_single_piece`` never emits a special id; those ids are printed
too, so the test can show the literal match is not the BPE segmentation.

The Rust tests hard-code this script's output. Re-run it to regenerate;
do not type token ids by hand.

    python3 scripts/gen_cl100k_pieces.py
"""

from __future__ import annotations

import tiktoken

# Pieces the A3 test encodes as one pre-split chunk.
PIECES = [
    "hello",
    " world",
    "🦀",
    "   ",
    "\n\n",
    "\t",
    "\r\n",
    "12345",
    "999999",
    "0",
    "café",
    "日本語",
    "a",
    "  hello",
    "🙂",
    "é",
    "Привет",
    "你好",
    "tokenization",
    "\n",
    "aaaaaaaa",
    "...",
    "👨‍👩‍👧‍👦",
    " ",
    "hello world",
    "ABC",
    "\t\t\t",
    "café déjà",
    # Longer than 100 bytes, so tiktoken's Rust encoder uses its heap path.
    # These still have to match the scan. Ids below are _encode_single_piece.
    "a" * 120,
    "The quick brown fox jumps over the lazy dog. " * 4,
]

SPECIALS = [
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|endofprompt|>",
]


def rust_string(text: str) -> str:
    out = ['"']
    for ch in text:
        code = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif code < 0x20 or code == 0x7F:
            out.append(f"\\u{{{code:x}}}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def rust_ids(ids: list[int]) -> str:
    return "&[" + ", ".join(str(i) for i in ids) + "]"


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"// tiktoken {tiktoken.__version__} cl100k_base")
    print(f"// mergeable tokens: {len(enc._mergeable_ranks)}")
    print("// ordinary pieces: Encoding._encode_single_piece (no regex)")
    print("const PIECES: &[(&str, &[u32])] = &[")
    for piece in PIECES:
        piece_ids = enc._encode_single_piece(piece)
        full_ids = enc.encode(piece, disallowed_special=())
        note = ""
        if piece_ids != full_ids:
            note = f"  // encode() with the regex is {full_ids}"
        print(f"    ({rust_string(piece)}, {rust_ids(piece_ids)}),{note}")
    print("];")
    print()
    print("// specials: Encoding.encode(..., allowed_special={token})")
    print("const SPECIALS: &[(&str, &[u32])] = &[")
    for token in SPECIALS:
        ids = enc.encode(token, allowed_special={token})
        print(f"    ({rust_string(token)}, {rust_ids(ids)}),")
    print("];")
    print()
    print("// same strings through _encode_single_piece, which ignores specials")
    print("const SPECIALS_AS_BPE: &[(&str, &[u32])] = &[")
    for token in SPECIALS:
        ids = enc._encode_single_piece(token)
        print(f"    ({rust_string(token)}, {rust_ids(ids)}),")
    print("];")


if __name__ == "__main__":
    main()
