"""Parity with tiktoken 0.14.0 on the A4 fixtures, plus batch and errors.

Requires the cl100k vocab cache (`scripts/fetch-cl100k.sh`) and:

    pip install 'tiktoken==0.14.0' pytest
    maturin develop --release
"""

from __future__ import annotations

import json
import pathlib
import threading
import time

import pytest
import tiktoken

import mahad_tok

FIXTURES = (
    pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "cl100k_cases.jsonl"
)


def load_cases() -> list[dict]:
    cases = []
    with FIXTURES.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


@pytest.fixture(scope="session")
def enc():
    return mahad_tok.Encoding.cl100k_base()


@pytest.fixture(scope="session")
def tik():
    if tiktoken.__version__ != "0.14.0":
        raise AssertionError(f"need tiktoken 0.14.0, found {tiktoken.__version__}")
    return tiktoken.get_encoding("cl100k_base")


def call_encode(encoder, text: str, specials):
    if specials == "all":
        return encoder.encode(text, allowed_special="all")
    return encoder.encode(text, allowed_special=set(specials))


def test_fixtures_match_tiktoken_encode_and_round_trip(enc, tik):
    cases = load_cases()
    assert len(cases) >= 200, len(cases)
    saw_disallowed = False
    for idx, case in enumerate(cases):
        text = case["text"]
        specials = case["specials"]
        if "disallowed" in case:
            saw_disallowed = True
            with pytest.raises(ValueError):
                call_encode(enc, text, specials)
            continue
        got = call_encode(enc, text, specials)
        expected = call_encode(tik, text, specials)
        assert got == expected, f"case {idx} {text!r}"
        assert got == case["ids"], f"case {idx} fixture ids"
        assert enc.decode(got) == text, f"case {idx} round trip"
        # Default encode (empty allowed set) agrees with tiktoken wherever
        # the text does not contain a disallowed special.
        if specials == "all" and "<|" not in text:
            assert enc.encode(text) == tik.encode(text), f"case {idx} default encode"
    assert saw_disallowed


def test_disallowed_special_is_value_error(enc):
    with pytest.raises(ValueError, match="endoftext"):
        enc.encode("hello <|endoftext|>")
    with pytest.raises(ValueError, match="endoftext"):
        enc.encode_batch(["ok", "hello <|endoftext|>"], num_threads=2)


def test_unknown_id_is_value_error(enc):
    with pytest.raises(ValueError, match="unknown token"):
        enc.decode([99_999_999])


def test_encode_batch_matches_tiktoken(enc, tik):
    cases = load_cases()
    texts = [
        case["text"]
        for case in cases
        if "disallowed" not in case and case["specials"] == "all"
    ]
    assert texts
    got = enc.encode_batch(texts, num_threads=4, allowed_special="all")
    expected = tik.encode_batch(texts, num_threads=4, allowed_special="all")
    assert got == expected
    one = enc.encode_batch(texts, num_threads=1, allowed_special="all")
    assert one == [enc.encode(text, allowed_special="all") for text in texts]
    assert enc.encode_batch([]) == []
    with pytest.raises(ValueError):
        enc.encode_batch(["hi"], num_threads=0)


def _best(fn, repeats: int = 3) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return min(samples)


def test_encode_batch_8_threads_beats_single_thread(enc):
    # ~120 chars * 500 * 64 texts is a few megabytes. Large enough that the
    # BPE work dominates the Python call, small enough for a unit test.
    # This machine may have fewer than 8 cores; 8 workers should still beat
    # one worker when the GIL is released.
    sentence = (
        "It is a truth universally acknowledged, that a single man in "
        "possession of a good fortune, must be in want of a wife. "
    )
    texts = [sentence * 500] * 64
    enc.encode_batch(texts[:2], num_threads=1, allowed_special="all")
    enc.encode_batch(texts[:2], num_threads=8, allowed_special="all")

    def loop():
        for text in texts:
            enc.encode(text, allowed_special="all")

    t_loop = _best(loop)
    t1 = _best(lambda: enc.encode_batch(texts, num_threads=1, allowed_special="all"))
    t8 = _best(lambda: enc.encode_batch(texts, num_threads=8, allowed_special="all"))
    print(
        f"batch 8 threads {t8:.3f}s, batch 1 thread {t1:.3f}s, "
        f"python encode loop {t_loop:.3f}s"
    )
    assert t8 < t1, (t8, t1, t_loop)
    assert t8 < t_loop, (t8, t_loop)


def test_encode_batch_releases_the_gil(enc):
    sentence = "The quick brown fox jumps over the lazy dog. "
    # About twice the 8-thread speed test, so the call stays long enough
    # for the watcher even when four cores are busy.
    texts = [sentence * 4000] * 80
    stop = threading.Event()
    ticks: list[float] = []

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.01)

    thread = threading.Thread(target=ticker)
    thread.start()
    time.sleep(0.05)
    before = len(ticks)
    start = time.perf_counter()
    enc.encode_batch(texts, num_threads=4, allowed_special="all")
    elapsed = time.perf_counter() - start
    during = len(ticks) - before
    stop.set()
    thread.join()
    print(f"gil watcher ticks during encode_batch: {during} in {elapsed:.3f}s")
    assert elapsed > 0.1, elapsed
    # sleep(0.01) only finishes its Python bytecodes if this thread can take
    # the GIL while encode_batch is still running. Holding the GIL for the
    # whole call leaves `during` at 0.
    assert during >= 5, (during, elapsed)
