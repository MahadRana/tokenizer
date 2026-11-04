#!/usr/bin/env python3
"""Write a gitignored ~100 MB corpus by repeating public-domain Gutenberg texts.

The files are the same Project Gutenberg ids as ``scripts/cross_check_tiktoken.py``.
They are downloaded into a temporary directory, concatenated, and repeated
until the UTF-8 byte length is at least 100_000_000 (decimal MB, the unit
``BENCHMARKS.md`` uses for MB/s). The output is ``bench-corpus.txt`` in the
repo root unless ``--out`` is set. Nothing is committed.

    python3 scripts/gen_bench_corpus.py
    python3 scripts/gen_bench_corpus.py --out /tmp/bench-corpus.txt

``--book book.txt`` repeats that one file instead of downloading the set.
Use it when Gutenberg is unreachable. Say so in the benchmark notes.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET_BYTES = 100_000_000

# Same ids as scripts/cross_check_tiktoken.py. Sizes vary; the script repeats
# whatever it actually downloaded until TARGET_BYTES.
GUTENBERG = [
    (1342, "Pride and Prejudice (English)"),
    (2701, "Moby-Dick (English)"),
    (4650, "Candide (French)"),
    (17489, "Les misérables, Tome I (French)"),
    (13951, "Les trois mousquetaires (French)"),
    (2229, "Faust (German)"),
    (2000, "Don Quijote (Spanish)"),
    (1012, "La Divina Commedia (Italian)"),
    (3333, "Os Lusíadas (Portuguese)"),
    (23841, "漢書 (Chinese)"),
]

USER_AGENT = "tokenizer-a6-bench/0.1 (educational; public-domain texts)"


def fetch(dest: pathlib.Path) -> str:
    parts: list[str] = []
    for ebook_id, title in GUTENBERG:
        url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        path = dest / f"pg{ebook_id}.txt"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
        except Exception as err:
            raise SystemExit(f"download {url} ({title}) failed: {err}") from err
        path.write_bytes(data)
        text = data.decode("utf-8")
        print(f"fetched {title}: {len(data)} bytes", file=sys.stderr)
        parts.append(text)
    return "".join(parts)


def repeat_to_target(piece: str) -> str:
    if piece == "":
        raise SystemExit("corpus piece is empty")
    piece_bytes = len(piece.encode("utf-8"))
    copies = (TARGET_BYTES + piece_bytes - 1) // piece_bytes
    return piece * copies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "bench-corpus.txt",
    )
    parser.add_argument(
        "--book",
        type=pathlib.Path,
        help="repeat this UTF-8 file instead of downloading Gutenberg texts",
    )
    args = parser.parse_args()
    if args.book is not None:
        piece = args.book.read_bytes().decode("utf-8")
        print(f"repeating {args.book} ({len(piece.encode())} bytes)", file=sys.stderr)
    else:
        with tempfile.TemporaryDirectory(prefix="tokenizer-a6-") as tmp:
            piece = fetch(pathlib.Path(tmp))
    text = repeat_to_target(piece)
    data = text.encode("utf-8")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(data)
    print(f"wrote {args.out} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
