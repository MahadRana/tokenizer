#!/usr/bin/env python3
"""Single-thread and batch throughput against tiktoken 0.14.0.

Times ``Encoding.encode`` on the whole text, then ``encode_batch`` at 1, 4,
and 8 threads on the same text split into chunks of ``CHUNK_CHARS`` Python
characters. tiktoken is called the same way (``allowed_special="all"``).
MB/s uses decimal megabytes (bytes / 1e6 / seconds), the same unit as the
A4 note in ``BENCHMARKS.md``.

The batch is not the same work as one ``encode`` of the whole string: a
chunk boundary is a hard cut, so pairs do not merge across it. The script
checks that each timed call matches tiktoken on that same input.

Needs ``book.txt``, ``bench-corpus.txt`` (``scripts/gen_bench_corpus.py``),
the cl100k cache, and ``maturin develop --release`` so ``mahad_tok`` imports.

    python3 scripts/bench_vs_tiktoken.py
"""

from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path

import tiktoken

import mahad_tok

ROOT = Path(__file__).resolve().parents[1]
CHUNK_CHARS = 32_768
# Book is short, so repeat the timing. The 100 MB file is one or two passes.
BOOK_REPEATS = 5
CORPUS_REPEATS = 2


def require_tiktoken():
    if tiktoken.__version__ != "0.14.0":
        raise SystemExit(
            f"need tiktoken 0.14.0, found {tiktoken.__version__}; refusing to compare"
        )
    return tiktoken.get_encoding("cl100k_base")


def read_text(path: Path) -> str:
    # Keep CR LF. Universal newlines would make the two sides encode
    # different strings if a caller rewrote the file.
    with path.open(encoding="utf-8", newline="") as fh:
        return fh.read()


def split_chars(text: str, chars: int) -> list[str]:
    return [text[i : i + chars] for i in range(0, len(text), chars)] or [""]


def mb_per_s(nbytes: int, seconds: float) -> float:
    return (nbytes / 1e6) / seconds


def time_call(fn, repeats: int) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return samples


def mean(samples: list[float]) -> float:
    return sum(samples) / len(samples)


def check_equal(label: str, got, expected) -> None:
    if got == expected:
        return
    if isinstance(got, list) and got and isinstance(got[0], list):
        for idx, (a, b) in enumerate(zip(got, expected)):
            if a != b:
                raise SystemExit(f"{label}: chunk {idx} mismatch (lens {len(a)} vs {len(b)})")
        raise SystemExit(
            f"{label}: length mismatch {len(got)} vs {len(expected)}"
        )
    raise SystemExit(f"{label}: mismatch (lens {len(got)} vs {len(expected)})")


def bench_one(label: str, text: str, ours, tik, repeats: int) -> None:
    nbytes = len(text.encode("utf-8"))
    chunks = split_chars(text, CHUNK_CHARS)
    print(
        f"\n{label}: {nbytes} bytes, {len(chunks)} chunks of {CHUNK_CHARS} chars",
        file=sys.stderr,
    )

    got = ours.encode(text, allowed_special="all")
    expected = tik.encode(text, allowed_special="all")
    check_equal(f"{label} single-thread", got, expected)
    print(f"{label} single-thread tokens {len(got)}", file=sys.stderr)

    for threads in (1, 4, 8):
        got_b = ours.encode_batch(chunks, num_threads=threads, allowed_special="all")
        exp_b = tik.encode_batch(chunks, num_threads=threads, allowed_special="all")
        check_equal(f"{label} batch {threads}", got_b, exp_b)

    def our_single():
        ours.encode(text, allowed_special="all")

    def tik_single():
        tik.encode(text, allowed_special="all")

    # Warmups are outside the samples. The rayon pool for 8 threads is
    # created on the first encode_batch, which this warmup pays for.
    our_single()
    tik_single()
    ours.encode_batch(chunks, num_threads=8, allowed_special="all")
    tik.encode_batch(chunks, num_threads=8, allowed_special="all")

    rows = [("single-thread", time_call(our_single, repeats), time_call(tik_single, repeats))]
    for threads in (1, 4, 8):
        rows.append(
            (
                f"batch {threads} threads",
                time_call(
                    lambda threads=threads: ours.encode_batch(
                        chunks, num_threads=threads, allowed_special="all"
                    ),
                    repeats,
                ),
                time_call(
                    lambda threads=threads: tik.encode_batch(
                        chunks, num_threads=threads, allowed_special="all"
                    ),
                    repeats,
                ),
            )
        )

    print(f"\n### {label}")
    print()
    print("| | mahad_tok | tiktoken 0.14.0 | mahad / tiktoken |")
    print("|---|---|---|---|")
    for name, our_samples, tik_samples in rows:
        our_mean = mean(our_samples)
        tik_mean = mean(tik_samples)
        ratio = our_mean / tik_mean
        print(
            f"| {name} | {mb_per_s(nbytes, our_mean):.3f} MB/s "
            f"({our_mean:.3f} s) | {mb_per_s(nbytes, tik_mean):.3f} MB/s "
            f"({tik_mean:.3f} s) | {ratio:.3f}× |"
        )
        print(
            f"  samples {name}: ours {[round(s, 4) for s in our_samples]} "
            f"tiktoken {[round(s, 4) for s in tik_samples]}",
            file=sys.stderr,
        )


def machine_specs() -> None:
    cpu = "unknown"
    mem = "unknown"
    cores: int | None = None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
        for line in meminfo.splitlines():
            if line.startswith("MemTotal"):
                mem = line.split(":", 1)[1].strip()
                break
        cores = sum(1 for line in cpuinfo.splitlines() if line.startswith("processor"))
    except OSError as err:
        print(f"procfs: {err}", file=sys.stderr)
    rustc = subprocess.run(
        ["rustc", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    print("### Machine")
    print()
    print(f"- OS: {platform.platform()} ({platform.machine()})")
    print(f"- CPU: {cpu}")
    print(f"- logical CPUs: {cores if cores is not None else 'unknown'}")
    print(f"- RAM: {mem}")
    print(f"- rustc: {rustc.stdout.strip() or rustc.stderr.strip()}")
    print(f"- python: {sys.version.split()[0]} ({sys.executable})")
    print(f"- tiktoken: {tiktoken.__version__}")
    if cores is not None and cores < 8:
        print(
            f"- This machine has {cores} logical CPUs, fewer than 8. "
            "The 8-thread rows were still run; they oversubscribe the cores."
        )


def main() -> None:
    book = ROOT / "book.txt"
    corpus = ROOT / "bench-corpus.txt"
    for path in (book, corpus):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    ours = mahad_tok.Encoding.cl100k_base()
    tik = require_tiktoken()
    machine_specs()
    bench_one("book", read_text(book), ours, tik, BOOK_REPEATS)
    bench_one("100 MB corpus", read_text(corpus), ours, tik, CORPUS_REPEATS)


if __name__ == "__main__":
    main()
