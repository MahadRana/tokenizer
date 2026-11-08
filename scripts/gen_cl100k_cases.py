#!/usr/bin/env python3
"""Generate tests/fixtures/cl100k_cases.jsonl from Python tiktoken.

Oracle: tiktoken 0.14.0, encoding ``cl100k_base``. Token ids are whatever
``Encoding.encode`` returns. Do not hand-edit the jsonl and do not invent
ids. Re-run this script.

The pattern asserted below is ``cl100k_base()["pat_str"]`` from
https://github.com/openai/tiktoken/blob/3591ff175d6a80efbe4fcc7f0e219ddd4b8c52f1/tiktoken_ext/openai_public.py
tiktoken 0.14.0 stores that same string on ``Encoding._pat_str``.

    python3 scripts/gen_cl100k_cases.py
    python3 scripts/gen_cl100k_cases.py --book book.txt --ids-out /tmp/ids.bin

``--book`` writes little-endian uint32 ids for
``encode(text, allowed_special="all")`` and checks that the default
``encode(text)`` (disallowed specials are an error) agrees. Pride and
Prejudice has none of the cl100k specials, so the two calls match.
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys

import tiktoken

# Verbatim pat_str from the commit above. A mismatch means this tiktoken
# is not the oracle the Rust regex was copied from.
CL100K_PAT = (
    r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+|"
    r" ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "cl100k_cases.jsonl"


def read_text_keep_newlines(path: pathlib.Path) -> str:
    """Read UTF-8 without translating ``\\r\\n`` to ``\\n``.

    ``Path.read_text`` and ``open`` default to universal newlines, which
    would make the Python oracle encode a different string than Rust's
    ``read_to_string``.
    """
    with path.open(encoding="utf-8", newline="") as fh:
        return fh.read()

SPECIALS = [
    "<|endoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|endofprompt|>",
]


def load_encoder():
    if tiktoken.__version__ != "0.14.0":
        raise SystemExit(
            f"need tiktoken 0.14.0, found {tiktoken.__version__}; refusing to invent ids"
        )
    enc = tiktoken.get_encoding("cl100k_base")
    if enc._pat_str != CL100K_PAT:
        raise SystemExit(
            "cl100k pat_str does not match openai_public.py at 3591ff1; refusing to invent ids\n"
            f"tiktoken: {enc._pat_str!r}"
        )
    return enc


def parse_disallowed(exc: ValueError) -> str:
    prefix = "Encountered text corresponding to disallowed special token "
    msg = str(exc)
    if not msg.startswith(prefix):
        raise SystemExit(f"unexpected tiktoken error: {msg}")
    rest = msg[len(prefix) :]
    end = rest.find(".\n")
    if end < 0:
        raise SystemExit(f"could not read the special token out of: {msg}")
    return ast.literal_eval(rest[:end])


class Cases:
    def __init__(self, enc) -> None:
        self.enc = enc
        self.rows: list[dict] = []
        self.seen: set[tuple] = set()

    def add_all(self, text: str) -> None:
        ids = self.enc.encode(text, allowed_special="all")
        self._push({"text": text, "specials": "all", "ids": ids})

    def add_subset(self, text: str, allowed: list[str]) -> None:
        try:
            ids = self.enc.encode(text, allowed_special=set(allowed))
        except ValueError as exc:
            token = parse_disallowed(exc)
            self._push({"text": text, "specials": list(allowed), "disallowed": token})
        else:
            self._push({"text": text, "specials": list(allowed), "ids": ids})

    def _push(self, row: dict) -> None:
        specials = row["specials"]
        key = (
            row["text"],
            "all" if specials == "all" else tuple(specials),
            row.get("disallowed"),
        )
        if key in self.seen:
            return
        self.seen.add(key)
        self.rows.append(row)


def fill(cases: Cases) -> None:
    contractions = [
        "'s",
        "'S",
        "'t",
        "'T",
        "'re",
        "'RE",
        "'ve",
        "'VE",
        "'m",
        "'M",
        "'ll",
        "'LL",
        "'d",
        "'D",
        "it's",
        "It's",
        "IT'S",
        "that's",
        "John's",
        "don't",
        "Don't",
        "can't",
        "won't",
        "I'll",
        "we'll",
        "they'll",
        "I'LL",
        "we'll go",
        "I've",
        "you've",
        "they've",
        "I'm",
        "you're",
        "we're",
        "I'd",
        "you'd",
        "we'd",
        "it's John's book and he'll read it",
        "the dog's bone",
        "friends'",
        "rock 'n' roll",
        "y'all",
        "''s",
    ]
    for text in contractions:
        cases.add_all(text)

    for n in range(1, 9):
        cases.add_all(" " * n)
        cases.add_all("\t" * n)
        cases.add_all("\n" * n)
        cases.add_all("\r\n" * n)
    whitespace = [
        "\r",
        "\r\r",
        "\n\r",
        "\n\r\n",
        " \t",
        "\t ",
        " \n",
        "\n ",
        "\t\n",
        "\n\t",
        "a b",
        "a  b",
        "a   b",
        "a\tb",
        "a\t\tb",
        "a\nb",
        "a\n\nb",
        "a\r\nb",
        "hello\r\nworld",
        "hello   ",
        "   hello",
        " \n \n ",
        "end of line \n",
        "trailing spaces   \n",
        "mix \t\r\n\t end",
        "\u00a0",
        "a\u00a0b",
        "line1\r\nline2\r\n",
    ]
    for text in whitespace:
        cases.add_all(text)

    for n in range(0, 40):
        cases.add_all(str(n))
    for n in (99, 100, 101, 999, 1000, 1001, 12345, 123456, 123456789, 123456789012):
        cases.add_all(str(n))
    for text in (
        "123",
        "1234",
        "123456789",
        "999999",
        "000",
        "0000",
        "00000",
        "12345 678 9 00000",
        "id=12345",
        "abc12345def",
        " 12345",
        "12345 ",
        "price 12345",
    ):
        cases.add_all(text)

    emoji = [
        "🦀",
        "🙂",
        "😀",
        "👍",
        "👍🏻",
        "👨‍👩‍👧‍👦",
        "🇺🇸",
        "🏳️‍🌈",
        "👩‍💻",
        "hello 🦀",
        "🦀🦀",
        "a🙂b",
        "\u200d",
        "👨\u200d👩\u200d👧\u200d👦",
        "☺️",
        "1️⃣",
        "🇯🇵",
        "e\u0301🙂",
    ]
    for text in emoji:
        cases.add_all(text)

    cjk = [
        "日本語",
        "你好",
        "こんにちは",
        "한국어",
        "中文",
        "漢字",
        "カタカナ",
        "ひらがな",
        "hello日本語",
        "日本語hello",
        "价格是123元",
        "你好，世界",
        "こんにちは世界",
        "「こんにちは」",
        "《中文》",
        "가나다",
        "한글",
        "中文\n日本語",
        "ＡＢＣ１２３",
    ]
    for text in cjk:
        cases.add_all(text)

    combining = [
        "e\u0301",
        "a\u0300",
        "n\u0303",
        "o\u0308",
        "c\u0327",
        "café",
        "cafe\u0301",
        "e\u0301\u0301",
        "A\u030a",
        "o\u0308e",
        "na\u0308ive",
        "Z\u030c",
        "a\u0308b\u0303c\u0327",
        "\u0301",
        "e\u0301\n",
    ]
    for text in combining:
        cases.add_all(text)

    ordinary = [
        "",
        "a",
        "hello",
        " world",
        "hello world",
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "  hello",
        "tokenization",
        "ABC",
        "...",
        "aaaaaaaa",
        "a" * 120,
        'fn main() { println!("hi"); }',
        "https://example.com/a?b=1&c=2",
        "user@example.com",
        "price is $12.50, or 12,50 €",
        "1 + 2 = 3",
        "foo\tbar\tbaz",
        "line1\nline2\nline3",
        '"quotes"',
        "it's a test of 'LL and 's",
        "🙂日本語",
        "  \n\t12345  ",
        "Mr. Bennet",
        "café déjà",
    ]
    for text in ordinary:
        cases.add_all(text)

    for token in SPECIALS:
        cases.add_all(token)
        cases.add_all(f"hello {token} world")
        cases.add_subset(token, [token])
        cases.add_subset(token, [])
        cases.add_subset(f"before {token} after", [token])
        cases.add_subset(f"before {token} after", [])
    cases.add_all("<|endoftext|><|fim_prefix|>")
    cases.add_all(" <|endoftext|> <|fim_middle|> <|fim_suffix|> ")
    cases.add_all("<|endoftext|> <|endoftext|>")
    cases.add_subset("<|endoftext|> and <|fim_prefix|>", ["<|endoftext|>"])
    cases.add_subset("<|endoftext|> and <|fim_prefix|>", ["<|fim_prefix|>"])
    cases.add_subset(
        "<|endoftext|> and <|fim_prefix|>",
        ["<|endoftext|>", "<|fim_prefix|>"],
    )
    cases.add_subset("no specials here", [])
    cases.add_subset("no specials here", ["<|endoftext|>"])
    cases.add_subset("<|endoftext>", [])
    cases.add_all("<|endoftext>")
    cases.add_all("endoftext")
    cases.add_subset("x<|endoftext|>y<|fim_suffix|>z", ["<|endoftext|>"])
    cases.add_subset(
        "x<|endoftext|>y<|fim_suffix|>z",
        ["<|endoftext|>", "<|fim_suffix|>"],
    )

    book_path = ROOT / "book.txt"
    if not book_path.is_file():
        raise SystemExit(f"missing {book_path}; the fixture includes excerpts from it")
    book = read_text_keep_newlines(book_path)
    for start, length in (
        (0, 180),
        (800, 240),
        (4_000, 300),
        (20_000, 220),
        (80_000, 260),
        (200_000, 240),
        (500_000, 300),
        (700_000, 200),
    ):
        cases.add_all(book[start : start + length])
    kept = 0
    for line in book.splitlines():
        if "'" in line and 30 <= len(line) <= 160:
            cases.add_all(line)
            kept += 1
            if kept >= 12:
                break


def write_fixtures(cases: Cases) -> None:
    if len(cases.rows) < 200:
        raise SystemExit(f"only {len(cases.rows)} cases; need at least 200")
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE.open("w", encoding="utf-8") as fh:
        for row in cases.rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases.rows)} cases to {FIXTURE}")


def dump_book(enc, book: str, ids_out: str) -> None:
    text = read_text_keep_newlines(pathlib.Path(book))
    ids = enc.encode(text, allowed_special="all")
    # Default encode errors if a disallowed special is present. The book
    # must not contain one: our `encode` allows every special, and the test
    # compares against allowed_special="all", which equals the default call
    # only when no special occurs.
    default = enc.encode(text)
    if ids != default:
        raise SystemExit("book contains a special token; default encode disagrees")
    import array

    arr = array.array("I", ids)
    if arr.itemsize != 4 or sys.byteorder != "little":
        raise SystemExit(f"unexpected endian layout itemsize={arr.itemsize} order={sys.byteorder}")
    pathlib.Path(ids_out).write_bytes(arr.tobytes())
    print(
        f"book bytes {len(text.encode('utf-8'))} tokens {len(ids)}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--book", help="text file to encode instead of writing the fixture")
    parser.add_argument("--ids-out", help="little-endian uint32 ids, with --book")
    args = parser.parse_args()
    enc = load_encoder()
    if args.book or args.ids_out:
        if not args.book or not args.ids_out:
            raise SystemExit("--book and --ids-out are both required")
        dump_book(enc, args.book, args.ids_out)
        return
    cases = Cases(enc)
    fill(cases)
    write_fixtures(cases)


if __name__ == "__main__":
    main()
