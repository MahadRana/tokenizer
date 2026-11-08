# Benchmarks

Same command for every row:

```
hyperfine --warmup 2 'target/release/tokenizer book.txt "hello"'
```

Corpus: Pride and Prejudice, Project Gutenberg #1342, saved locally as `book.txt` (gitignored) from <https://www.gutenberg.org/cache/epub/1342/pg1342.txt>. 772,386 bytes. Every run below learned all 1,000 requested merges. The probe `"hello"` encodes to 3 tokens (`[361, 285, 111]`); that probe is not the compression ratio.

Compression: **3.239 bytes/token** (772,386 / 238,459). The token count is the length of the training sequence after those 1,000 merges. A2's fast trainer matches the A1 merges on this file (`fast_training_matches_naive_on_book`), so the ratio did not change.

## A1 remeasure and A2, same machine

Measured 2026-11-01 with hyperfine 2.0.0 and rustc 1.99.1 (58cf8f9c3 2026-10-13). The A1 number is a fresh release build of `3304b21` (`feat/a1-train-test`) taken immediately before the A2 trainer landed. The A2 number is a release build of `626530e`.

| | Wall time | Runs |
|---|---|---|
| A1, recount every merge | **4.173 s ± 0.048 s** | 10 |
| A2, incremental counts + lazy heap | **149.1 ms ± 5.6 ms** | 20 |

Speedup against that remeasure: 4.173 s / 0.1491 s = **28.0×**. Hyperfine raised the A2 run count from 10 to 20 on its own because the command finished quickly. Means from the raw summaries: A1 4.172626 s (σ 0.047678 s, min 4.142 s, max 4.302 s); A2 0.149084 s (σ 0.005563 s, min 0.142 s, max 0.163 s).

Machine (both rows):

- Linux 6.12.94+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports the generic name "Intel(R) Xeon(R) Processor")
- MemTotal 16,387,280 kB

No criterion benches. The timed program is training plus encoding the five-byte probe, and training is the whole cost.

## A1: earlier published baseline

Measured 2026-11-01 on commit `15f5bc2b00ec2fe25cab90ac24788f31f8a2f6ff`, same hyperfine invocation, same class of machine and the same rustc. Wall time **4.206 s ± 0.036 s** (mean 4.206192 s, σ 0.036482 s, min 4.171 s, max 4.273 s). Kept so the original write-up is still here. The 28× figure above uses the remeasure, not this row: it was taken in a different session, a few tens of milliseconds off this machine's rerun.

## A3: cl100k vocab round trip

Not the training command. `decode_encode_piece_round_trips_every_vocab_entry` encodes every mergeable cl100k token and checks `decode_bytes` returns those bytes. Two release runs on 2026-11-01, rustc 1.99.1 (58cf8f9c3 2026-10-13), same machine as the A2 rows: **53.056 ms** and **53.227 ms** for 100,256 tokens. Debug on the test profile was 337 ms. The test fails the run if release exceeds 1 s. These are two clock readings, not a hyperfine mean.

A release run on 2026-11-02, same machine and the same rustc, printed **59.284 ms**. The timer starts after `load_cl100k` returns, so the new regex compile is not in that number. The piece scan is unchanged. Still under the 1 s budget. Three readings (53.056, 53.227, 59.284 ms) are not a mean.

## A4: cl100k encode throughput on the book

Not the training command. `cargo bench --bench cl100k_encode` encodes Pride and Prejudice with `RankTokenizer::encode` (specials allowed, cl100k regex, then piece BPE). The vocab is the cached `cl100k_base.tiktoken`. Criterion 0.8.2, `Throughput::Bytes` of the file length.

Measured 2026-11-02, rustc 1.99.1 (58cf8f9c3 2026-10-13), same machine as the rows above (Linux 6.12.94+ x86_64, 4 vCPUs, Intel Xeon family 6 model 207, MemTotal 16,387,280 kB). 100 samples. Criterion warned that 100 samples wanted about 7.2 s and then collected them.

| | time | throughput |
|---|---|---|
| `cl100k_encode/book` | **73.636 ms** (72.820–74.501 ms) | **10.003 MiB/s** (9.8872–10.115 MiB/s) |

772,386 bytes / 0.073636 s = **10.49 MB/s** using 10^6 bytes per MB. Criterion's MiB is 2^20. The file encodes to 177,179 tokens (`encode_matches_tiktoken_on_book`, release), 4.36 bytes/token. Gnuplot was not installed; criterion used the plotters backend. The number is one criterion estimate, not a hyperfine mean of the training binary.

## A5: property tests, fuzz, and the Python cross-check

Not the training command. Same machine as the rows above (Linux 6.12.94+ x86_64, 4 vCPUs, Intel Xeon family 6 model 207, MemTotal 16,387,280 kB). Debug tests used rustc 1.99.1 (58cf8f9c3 2026-10-13). The fuzzer used rustc 1.101.0-nightly (1c3a65069 2026-11-02).

| | measured |
|---|---|
| `prop_round_trip_bytes`, 10,000 cases | **1.72 s** (harness; includes the cl100k load) |
| `prop_matches_tiktoken_rs`, 10,000 cases | **2.24 s** (harness; includes the cl100k load and tiktoken-rs init) |
| `fuzz_encode`, libFuzzer `-max_total_time=1800` | **1,801 s**, **1,257,627** executions, 698 exec/s average, no mismatch |
| Python tiktoken 0.14.0 on 10 Gutenberg files | **10,610,901** bytes, **0** diffs |

`PROPTEST_CASES=20` on the round-trip property finished in 0.19 s, and `PROPTEST_CASES=10000` finished in 1.71 s, which is the unset default. CI runs the 10,000-case default. These are single harness readings, not hyperfine means.
