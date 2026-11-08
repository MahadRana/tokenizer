#!/usr/bin/env python3
"""Compare RankTokenizer.encode to Python tiktoken 0.14.0 on mixed text.

The corpus is not committed. With no arguments the script downloads
public-domain Project Gutenberg books (several languages) into a temporary
directory, encodes each file with this crate and with tiktoken, and deletes
the downloads. ``--path`` reads a local file or directory instead and does
not download anything.

Newlines are kept as they are on disk (``newline=""``). Rust
``read_to_string`` does the same. Universal-newline translation makes the
two oracles encode different strings.

    python3 scripts/cross_check_tiktoken.py
    python3 scripts/cross_check_tiktoken.py --path book.txt

Prints ``bytes_compared`` and ``diffs``. Exits 0 only when every file matches.
"""

from __future__ import annotations

import argparse
import pathlib
import struct
import subprocess
import sys
import tempfile
import urllib.request

import tiktoken

ROOT = pathlib.Path(__file__).resolve().parents[1]
VOCAB = pathlib.Path.home() / ".cache" / "tokenizer" / "cl100k_base.tiktoken"
EXAMPLE = ROOT / "target" / "release" / "examples" / "cl100k_ids"

# cache/epub URLs. Sizes are what Gutenberg returned for Content-Length when
# this list was chosen (they sum to about 10 MB). The script reports the
# byte count of the files it actually read.
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

USER_AGENT = "tokenizer-a5-cross-check/0.1 (educational; public-domain texts)"


def require_tiktoken_014() -> tiktoken.Encoding:
    if tiktoken.__version__ != "0.14.0":
        raise SystemExit(
            f"need tiktoken 0.14.0, found {tiktoken.__version__}; refusing to compare"
        )
    return tiktoken.get_encoding("cl100k_base")


def read_text_keep_newlines(path: pathlib.Path) -> str:
    with path.open(encoding="utf-8", newline="") as fh:
        return fh.read()


def fetch_corpus(dest: pathlib.Path) -> list[pathlib.Path]:
    paths = []
    for ebook_id, title in GUTENBERG:
        url = f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"
        path = dest / f"pg{ebook_id}.txt"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                path.write_bytes(resp.read())
        except Exception as err:
            raise SystemExit(f"download {url} ({title}) failed: {err}") from err
        print(f"fetched {title}: {path.stat().st_size} bytes", file=sys.stderr)
        paths.append(path)
    return paths


def paths_from(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file())
        if not files:
            raise SystemExit(f"no files in {path}")
        return files
    raise SystemExit(f"not a file or directory: {path}")


def ensure_encoder() -> pathlib.Path:
    fetch = ROOT / "scripts" / "fetch-cl100k.sh"
    subprocess.run([str(fetch)], check=True, cwd=ROOT)
    if not VOCAB.is_file():
        raise SystemExit(f"vocab missing after fetch: {VOCAB}")
    subprocess.run(
        ["cargo", "build", "--release", "--example", "cl100k_ids"],
        check=True,
        cwd=ROOT,
    )
    if not EXAMPLE.is_file():
        raise SystemExit(f"missing example binary: {EXAMPLE}")
    return EXAMPLE


def rust_ids(binary: pathlib.Path, path: pathlib.Path) -> list[int]:
    proc = subprocess.run(
        [str(binary), str(VOCAB), str(path)],
        check=False,
        capture_output=True,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"cl100k_ids failed on {path} ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace')}"
        )
    blob = proc.stdout
    if len(blob) % 4 != 0:
        raise SystemExit(f"id stream length {len(blob)} is not a multiple of 4")
    return list(struct.unpack(f"<{len(blob) // 4}I", blob))


def compare(enc: tiktoken.Encoding, binary: pathlib.Path, path: pathlib.Path) -> tuple[int, int]:
    text = read_text_keep_newlines(path)
    expected = enc.encode(text, allowed_special="all")
    got = rust_ids(binary, path)
    if got != expected:
        idx = next(
            (i for i, (a, b) in enumerate(zip(got, expected)) if a != b),
            min(len(got), len(expected)),
        )
        print(
            f"DIFF {path} at token {idx}: rust {got[idx:idx + 4]} "
            f"tiktoken {expected[idx:idx + 4]} "
            f"(rust {len(got)} ids, tiktoken {len(expected)} ids, {len(text.encode('utf-8'))} bytes)",
            file=sys.stderr,
        )
        return len(text.encode("utf-8")), 1
    print(
        f"ok {path.name}: {len(text.encode('utf-8'))} bytes, {len(got)} tokens",
        file=sys.stderr,
    )
    return len(text.encode("utf-8")), 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=pathlib.Path,
        help="UTF-8 file or directory to compare instead of downloading Gutenberg",
    )
    args = parser.parse_args()
    enc = require_tiktoken_014()
    binary = ensure_encoder()

    if args.path is not None:
        files = paths_from(args.path)
        total_bytes, diffs = run(enc, binary, files)
    else:
        with tempfile.TemporaryDirectory(prefix="tokenizer-a5-") as tmp:
            files = fetch_corpus(pathlib.Path(tmp))
            total_bytes, diffs = run(enc, binary, files)

    print(f"bytes_compared={total_bytes}")
    print(f"files={len(files)}")
    print(f"diffs={diffs}")
    if diffs != 0:
        raise SystemExit(1)


def run(
    enc: tiktoken.Encoding, binary: pathlib.Path, files: list[pathlib.Path]
) -> tuple[int, int]:
    total_bytes = 0
    diffs = 0
    for path in files:
        nbytes, ndiff = compare(enc, binary, path)
        total_bytes += nbytes
        diffs += ndiff
    return total_bytes, diffs


if __name__ == "__main__":
    main()
