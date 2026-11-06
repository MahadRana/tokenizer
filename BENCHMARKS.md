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

## A6: Python bindings vs tiktoken 0.14.0

Not the training command. `scripts/bench_vs_tiktoken.py` times `mahad_tok.Encoding.encode` on the whole string, then `encode_batch` at 1, 4, and 8 threads. tiktoken 0.14.0 is called the same way, with `allowed_special="all"`. The script checks token ids against tiktoken before it starts the clock (book: 177,179 tokens, the same count as the A4 book test; corpus: 36,411,860 tokens). A mismatch would have exited. None did.

MB/s is bytes / 1e6 / seconds, the decimal megabyte used in the A4 note. The ratio column is wall time, ours divided by tiktoken. 1.0× is a tie. Above 1 means tiktoken finished sooner. Batch rows split the same text into chunks of 32,768 Python characters (24 chunks on the book, 2,631 on the corpus). A chunk boundary is a hard cut, so those rows are not the same work as one `encode` of the whole string. Means of 5 samples on the book and 2 on the corpus.

The corpus is `bench-corpus.txt` (gitignored), 106,109,010 bytes. `scripts/gen_bench_corpus.py` downloaded the ten Gutenberg files from the A5 cross-check (10,610,901 bytes, the same total A5 recorded), concatenated them, and repeated that blob until it passed 100,000,000 bytes. Ten copies is 106,109,010. The files are not committed.

Measured 2026-11-04, rustc 1.99.1 (58cf8f9c3 2026-10-13), Python 3.12.3, maturin 1.16.0, PyO3 0.30.1, rayon 1.12.0, tiktoken 0.14.0. Same machine as the rows above.

- Linux 6.12.94+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores.
- MemTotal 16,387,280 kB

This machine has 4 logical CPUs, fewer than 8. The 8-thread rows were still run. They oversubscribe the cores. On the book, 8 threads is slower than 4. On the corpus, 8 and 4 are about the same, and both are faster than 1.

### Book (772,386 bytes)

| | mahad_tok | tiktoken 0.14.0 | time ratio |
|---|---|---|---|
| single-thread | **10.800 MB/s** (0.072 s) | **20.150 MB/s** (0.038 s) | **1.866×** |
| batch, 1 thread | 10.704 MB/s (0.072 s) | 14.957 MB/s (0.052 s) | 1.397× |
| batch, 4 threads | 31.374 MB/s (0.025 s) | 49.890 MB/s (0.015 s) | 1.590× |
| batch, 8 threads | 27.430 MB/s (0.028 s) | 36.846 MB/s (0.021 s) | 1.343× |

Book samples, seconds. Ours then tiktoken.

- single-thread: 0.0715, 0.0714, 0.0718, 0.0717, 0.0713 and 0.0394, 0.0380, 0.0382, 0.0380, 0.0381
- batch 1: 0.0732, 0.0721, 0.0720, 0.0719, 0.0715 and 0.0531, 0.0521, 0.0508, 0.0512, 0.0510
- batch 4: 0.0246, 0.0243, 0.0249, 0.0243, 0.0250 and 0.0158, 0.0151, 0.0151, 0.0158, 0.0156
- batch 8: 0.0287, 0.0252, 0.0288, 0.0306, 0.0275 and 0.0198, 0.0238, 0.0200, 0.0214, 0.0198

Single-thread on the book is **1.87×** tiktoken's time. That misses the 1.5× target. The A4 criterion loop on this file, same machine and the same rustc, was 10.49 MB/s. The Python `encode` is 10.80 MB/s, so the binding is not where the time went. tiktoken's encoder is faster on this text. 8 threads still beats the 1-thread batch (0.028 s vs 0.072 s).

### 100 MB corpus (106,109,010 bytes)

| | mahad_tok | tiktoken 0.14.0 | time ratio |
|---|---|---|---|
| single-thread | **8.523 MB/s** (12.449 s) | **11.277 MB/s** (9.410 s) | **1.323×** |
| batch, 1 thread | 6.873 MB/s (15.439 s) | 9.150 MB/s (11.597 s) | 1.331× |
| batch, 4 threads | 22.802 MB/s (4.654 s) | 28.838 MB/s (3.679 s) | 1.265× |
| batch, 8 threads | 23.206 MB/s (4.573 s) | 23.339 MB/s (4.546 s) | 1.006× |

Corpus samples, seconds. Ours then tiktoken. Two samples, so this is a mean of a short run, not a hyperfine σ.

- single-thread: 11.5335, 13.3654 and 9.4981, 9.3211
- batch 1: 15.8578, 15.0196 and 12.6839, 10.5101
- batch 4: 4.7939, 4.5132 and 3.0548, 4.3040
- batch 8: 4.6754, 4.4697 and 4.3611, 4.7316

Single-thread on the corpus is **1.32×** tiktoken's time, inside the 1.5× target. The two ours samples differ by about 1.8 s, so the ratio is coarser than the book row. 8 threads is 15.439 / 4.573 = **3.4×** the 1-thread batch. It does not beat 4 threads by a meaningful amount: there are only 4 cores.

A separate pytest on a synthetic list (64 copies of a ~60 KB sentence, minimum of 3 runs) measured `encode_batch(..., num_threads=8)` at 0.124 s, the 1-thread batch at 0.302 s, and a Python loop of `encode` at 0.293 s. A watcher thread recorded 33 GIL acquisitions during a 0.400 s `encode_batch` (`time.sleep(0.01)` in that thread). Those are one pytest run, not the tables above.

## B1: Qwen2.5-0.5B greedy on CPU

Not the training command. `cargo run --release -p qwen-infer -- "The capital of France is"` encodes with `RankTokenizer::load_qwen`, then takes 20 greedy steps. Each step runs the full sequence. There is no KV cache. The clock starts after the weights are loaded and stops after the 20th id. The continuation was ` Paris. It is the largest city in Europe and the third largest city in the world. It is`, the same 20 ids as the transformers float32 eager dump.

Measured 2026-11-05, rustc 1.99.1 (58cf8f9c3 2026-10-13), candle-core 0.11.1, candle-nn 0.11.1, `DType::F32`, `Device::Cpu`. Same machine as the A6 rows.

- Linux 6.12.94+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores.
- MemTotal 16,387,280 kB

| run | wall time | tokens/sec |
|---|---|---|
| 1 | 3.4210 s | 5.8462 |
| 2 | 3.8330 s | 5.2179 |

Both runs were warm (the weights had already been read). Mean of the two wall times: (3.4210 + 3.8330) / 2 = 3.6270 s, which is 20 / 3.6270 = 5.5142 tokens/sec. Two samples, not a hyperfine mean. The release ignored test that checks the logits also finished its forward plus the 20 steps in 5.97 s, and that number includes loading the mapped weights.

The encoder was not changed for speed in this milestone. The A6 single-thread ratios still stand on this machine: 1.87× tiktoken on the book, 1.32× on the 100 MB corpus. Closing the book gap is a change to the rank scan or the hash, measured again against those two rows.

## B2: KV cache vs the no-cache loop, 256 new tokens

Not the training command. Both sides generate 256 greedy tokens (`temperature = 0`, no EOS, no stop strings) after the weights are already mapped. The clock does not include that load. Cached time is split into the prompt forward (prefill) and the 256 decode forwards. The no-cache loop is `greedy_tokens`: one full forward of the growing sequence per new token, the B1 loop. Decode tok/s is 256 / decode seconds. The speedup is that rate divided by the no-cache rate. One sample each, not a hyperfine mean. Greedy ids matched on both rows.

Measured 2026-11-06, branch `feat/b2-kv-cache-sampling`, rustc 1.99.1 (58cf8f9c3 2026-10-13), candle-core 0.11.1, candle-nn 0.11.1, `DType::F32`, `Device::Cpu`. This boot's `uname` is Linux 6.12.113+ (the B1 rows were written as 6.12.94+). Same CPU, slightly different MemTotal.

- Linux 6.12.113+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores.
- MemTotal 16,387,328 kB

| prompt | uncached | cached prefill | cached decode | prefill+decode | speedup |
|---|---|---|---|---|---|
| 5 tokens, `"The capital of France is"` | **120.3043 s**, **2.1279 tok/s** | 0.1207 s | **47.4393 s**, **5.3964 tok/s** | 47.5600 s, 5.3827 tok/s | **2.54×** |
| 516 tokens, the same sentence repeated | **514.4022 s**, **0.4977 tok/s** | 1.3928 s | **55.8705 s**, **4.5820 tok/s** | 57.2633 s, 4.4706 tok/s | **9.21×** |

The 5× bar is the 516-token prompt. The 5-token prompt does not clear it, and the reason is the matmul, not a broken cache. A no-cache step of length L is one batched gemm: the f32 weights are read once and applied to L rows. A cached step is the same read applied to 1 row. Single forwards of the real model, same process, were 0.1861 s at length 1, 0.1502 s at 8, 0.2342 s at 32, 0.3135 s at 64, 0.4842 s at 128, and 0.8238 s at 256. A cached decode step was 0.1875 s, the same as a length-1 forward, so the cache append is not what the 47 s is. From length 32 up the line is about 0.145 s plus 2.5 ms per extra token. Summing that from length 5 through 260 is 121 s, which is the 120.3043 s row. The fixed 0.145 s is streaming roughly 2 GB of weights. A 1 GB copy in this process moved at 24.5 GB/s counting both the read and the write, so a pure read of that working set is already near 0.15 s. Cutting the cached step in half is not available on this guest. The ratio crosses 5× once the prompt is long enough for the per-token term to dominate. At 516 prompt tokens it is 514.4022 / 55.8705 = 9.21×.

`Tensor::cat` copies the prefix every step. Stored K and V are `(1, 2, seq, 64)` f32, 24 layers, both tensors: `seq * 24` KB per step. At seq 256 that step copies 6.0 MB. The sum over 256 steps is `24 KB * 256 * 257 / 2` = 790 MB. At the copy rate above that is under a tenth of a second, against a 47 s decode. The copy is about a tenth of one step when `seq * 24 KB` is ~18 ms, around 9,000 tokens. That is the cost B5 is for. It is not visible at 256.

The default CLI on the 5-token prompt, 20 greedy tokens, EOS ids `[151645, 151643]`, did not hit EOS. Prefill 0.1334 s, decode 3.9059 s, 5.1205 tok/s. The text is the B1 continuation. Twenty tokens is too short for the ratio above to show up: decode is still one weight read per token.

The encoder was not changed. The A6 ratios are untouched.
