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

### Book encode after the piece fast path

The tables above are the 2026-11-04 measurement, before this change. Remeasured 2026-11-10 on `feat/a6-encode-perf`, rustc 1.99.1 (58cf8f9c3 2026-10-13), Python 3.12.3, maturin 1.16.0, tiktoken 0.14.0. Same machine class as the B6 rows.

- Linux 6.12.113+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores.
- MemTotal 16,387,328 kB

`scripts/bench_vs_tiktoken.py` on `book.txt` only (772,386 bytes, 177,179 tokens, ids checked against tiktoken before the clock). Means of 5. The corpus file was not in this run. MB/s is bytes / 1e6 / seconds. The ratio is our wall time divided by tiktoken.

A split of one release `encode` of the book, before the change: the cl100k regex `find_iter` was about 28 ms, and `encode_piece` on the 162,624 pieces was about 41 ms. 151,179 of those pieces (674,292 bytes) were already a single vocab token and cost about 34 ms. The other 11,445 pieces cost about 6 ms. No piece was longer than 67 bytes. The merge loop on tokens the vocab already contains was the gap. The change emits that id directly for a `load_cl100k` tokenizer. `encode_bpe` is unchanged, and the vocab round-trip test still runs the merge loop on every token. A parsed vocab that is not cl100k does not take the shortcut.

Before, same command, this machine:

| | mahad_tok | tiktoken 0.14.0 | time ratio |
|---|---|---|---|
| single-thread | **10.480 MB/s** (0.074 s) | **19.950 MB/s** (0.039 s) | **1.904×** |
| batch, 1 thread | 10.343 MB/s (0.075 s) | 14.451 MB/s (0.053 s) | 1.397× |
| batch, 4 threads | 31.073 MB/s (0.025 s) | 49.002 MB/s (0.016 s) | 1.577× |
| batch, 8 threads | 27.334 MB/s (0.028 s) | 40.234 MB/s (0.019 s) | 1.472× |

Before samples, seconds. Ours then tiktoken.

- single-thread: 0.0731, 0.0761, 0.0728, 0.0730, 0.0734 and 0.0387, 0.0378, 0.0379, 0.0416, 0.0375
- batch 1: 0.0753, 0.0738, 0.0743, 0.0743, 0.0757 and 0.0531, 0.0516, 0.0520, 0.0520, 0.0585
- batch 4: 0.0259, 0.0249, 0.0246, 0.0246, 0.0242 and 0.0165, 0.0155, 0.0156, 0.0153, 0.0159
- batch 8: 0.0333, 0.0290, 0.0254, 0.0271, 0.0264 and 0.0210, 0.0192, 0.0208, 0.0193, 0.0157

After, a quiet run of the same book timing, once the machine was otherwise idle:

| | mahad_tok | tiktoken 0.14.0 | time ratio |
|---|---|---|---|
| single-thread | **18.100 MB/s** (0.043 s) | **20.232 MB/s** (0.038 s) | **1.118×** |
| batch, 1 thread | 17.794 MB/s (0.043 s) | 15.095 MB/s (0.051 s) | 0.848× |
| batch, 4 threads | 44.573 MB/s (0.017 s) | 49.249 MB/s (0.016 s) | 1.105× |
| batch, 8 threads | 40.731 MB/s (0.019 s) | 38.553 MB/s (0.020 s) | 0.947× |

After samples, seconds. Ours then tiktoken.

- single-thread: 0.0424, 0.0420, 0.0448, 0.0419, 0.0423 and 0.0389, 0.0386, 0.0373, 0.0387, 0.0373
- batch 1: 0.0428, 0.0422, 0.0422, 0.0480, 0.0419 and 0.0532, 0.0510, 0.0504, 0.0511, 0.0502
- batch 4: 0.0179, 0.0175, 0.0168, 0.0169, 0.0176 and 0.0161, 0.0160, 0.0153, 0.0155, 0.0155
- batch 8: 0.0185, 0.0203, 0.0197, 0.0183, 0.0180 and 0.0214, 0.0202, 0.0205, 0.0189, 0.0192

An earlier quiet run of the same function was single-thread **18.250 MB/s** (0.042 s) against **19.772 MB/s** (0.039 s), **1.083×**. Samples: 0.0424, 0.0428, 0.0425, 0.0419, 0.0420 and 0.0399, 0.0390, 0.0390, 0.0389, 0.0386.

`scripts/bench_vs_tiktoken.py` with the corpus file present printed a book single-thread mean of **15.448 MB/s** (0.050 s) against **19.789 MB/s** (0.039 s), **1.281×**. The first of our five samples was 0.0644 s. The other four were 0.0494, 0.0457, 0.0452, 0.0454. That run is under 1.5× too. The quiet means above are the steady rate. 18.100 / 10.480 = 1.73× the earlier MB/s on this boot.

The same script invocation timed the new corpus. `scripts/gen_bench_corpus.py` downloaded the ten Gutenberg ids and wrote `bench-corpus.txt` at **106,108,750** bytes (the 2026-11-04 file was 106,109,010; this download is 260 bytes shorter). 36,411,750 tokens, checked against tiktoken before the clock. Means of 2. There is no before row for this exact file.

| | mahad_tok | tiktoken 0.14.0 | time ratio |
|---|---|---|---|
| single-thread | **10.784 MB/s** (9.839 s) | **14.681 MB/s** (7.228 s) | **1.361×** |
| batch, 1 thread | 10.439 MB/s (10.165 s) | 10.313 MB/s (10.289 s) | 0.988× |
| batch, 4 threads | 27.756 MB/s (3.823 s) | 31.601 MB/s (3.358 s) | 1.139× |
| batch, 8 threads | 26.628 MB/s (3.985 s) | 25.638 MB/s (4.139 s) | 0.963× |

Corpus samples, seconds. Ours then tiktoken.

- single-thread: 10.1769, 9.5020 and 7.2049, 7.2509
- batch 1: 10.2688, 10.0607 and 11.6628, 8.9157
- batch 4: 3.8518, 3.7942 and 2.6494, 4.0661
- batch 8: 4.0850, 3.8848 and 3.0475, 5.2301

Single-thread on the book is **1.118×** tiktoken on the quiet run, and **1.281×** on the script run that included the 0.0644 s sample. Both are inside the 1.5× target. The corpus single-thread row is **1.361×**, also inside. The corpus has more pieces that are not already one token, so the shortcut covers less of that file than it does of the book.

Criterion `cl100k_encode/book`, same binary, flags `--warm-up-time 1 --measurement-time 5 --sample-size 50`. Before: **70.242 ms** (69.746–70.861 ms), **10.487 MiB/s** (10.395–10.561). After: **39.244 ms** (38.917–39.604 ms), **18.770 MiB/s** (18.599–18.927). 772,386 / 0.039244 s = **19.68 MB/s** at 10^6 bytes per MB. The first criterion pass on this boot was taken while another compile was running and is not the before number.

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

## B3: server TTFT and decode, chat template, 20-token cap

Not the training command, and not a sustained-throughput run. `qwen-server` in release, one request at a time, temperature 0, after a 1-token warmup so the f32 weights are already faulted in. The client is the OpenAI Python SDK 3.46.0 pointed at `http://127.0.0.1:8080/v1`. TTFT is the client's clock from `create(stream=True)` to the first chunk whose delta has `content`. The engine lines are the prefill forward and the decode interval printed by the engine thread. One warmed short request each, not a hyperfine mean and not a load test.

Measured 2026-11-07, branch `feat/b3-openai-api`, rustc 1.99.1 (58cf8f9c3 2026-10-13), candle-core 0.11.1, candle-nn 0.11.1, axum 0.8.9, tokio 1.54.1, `DType::F32`, `Device::Cpu`. Same machine as the B2 rows.

- Linux 6.12.113+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores.
- MemTotal 16,387,328 kB

The prompt is a chat turn, not the raw string from B1. The template adds the default system line and `<|im_start|>assistant\n`. `"The capital of France is"` is **34** prompt tokens. `"Say hi"` is **31**.

| call | prompt tokens | completion tokens | finish | prefill | decode | client TTFT | text |
|---|---|---|---|---|---|---|---|
| warmup, `max_tokens` 1 | 34 | 1 | length | 0.2104 s | **0.0002 s** | | `The` |
| stream, stops on EOS | 34 | 8 | stop | 0.2075 s | **1.2975 s** | **0.2100 s** | `The capital of France is Paris.` |
| non-stream, same request | 34 | 8 | stop | 0.2118 s | 1.3037 s | | same text, `total_tokens` 42 |

8 / 1.2975 = **6.17 tok/s** over the decode interval. The last of those 8 is the EOS id (`<|im_end|>`, not decoded). The other 7 are the text. Client TTFT is the prefill: the first token is sampled from the prompt logits, and the loop does not run a forward after a token it will not continue from. The warmup's decode of 0.0002 s is that same choice with `max_tokens` 1. A B2 step was about 0.187 s. 1.2975 / 7 forwards is 0.185 s, the same weight read.

The ignored SDK test on `"Say hi"`, `max_tokens` 8, did not hit EOS. Three runs (SDK stream, raw SSE, non-stream) were prefill 0.2092 / 0.2082 / 0.2389 s and decode 1.2300 / 1.2538 / 1.2547 s, completion tokens 8, finish `length`.

The B2 table is unchanged. This row is one request through HTTP, not the 256-token cache comparison. The ignored B2 test was re-run on this boot after the unused tail forward was removed: prompt 516, uncached 546.0694 s (0.4688 tok/s), cached decode 57.7040 s (4.4364 tok/s), **9.46×**, same greedy ids. Both sides were slower than the 2026-11-06 row (uncached 514.4022 s, decode 55.8705 s, 9.21×). One dropped forward in 256 is not what moved the ratio. The table above stays the B2 measurement.

## B4: eight concurrent requests, one warmed sample

Not a sustained run, and not a percentile. One process, temperature 0, EOS off so each call runs all 16 tokens. A warmup request of the same shape runs first and is not in the table. The single row is the next request. The batch row is eight of those requests issued together: 8 × 16 = **128** completion tokens, wall time from the first send to the last body. Total tok/s is that token count divided by that wall time. The ratio is the batch rate divided by the single rate. Client text matched across the eight and the single call.

Measured 2026-11-08, branch `feat/b4-continuous-batching`, rustc 1.99.1 (58cf8f9c3 2026-10-13), candle-core 0.11.1, candle-nn 0.11.1, `DType::F32`, `Device::Cpu`. Same machine as the B2 and B3 rows.

- Linux 6.12.113+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores.
- MemTotal 16,387,328 kB

| run | completion tokens | wall | total tok/s |
|---|---|---|---|
| one request, after one warmup | 16 | **2.8346 s** | **5.6444** |
| 8 concurrent | 128 | **3.7229 s** | **34.3820** |

34.38 tok/s vs 5.64 tok/s is **6.09×** (34.3820 / 5.6444). Prefill of each prompt is still its own forward. The shared work is the decode steps.

A separate call, not folded into the ratio: one 12-token stream was already producing content, then a 1-token request was sent. Client TTFT for that short request was **0.5532 s**, text `Hello`. The long request was still running (`max_running` 2) and was cancelled at 4 completion tokens when the test dropped it. One sample, not a latency distribution.

A second sample after the slow-client send path landed is not a replacement. Same commands, same machine: single **2.9118 s** (**5.4949 tok/s**), batch **3.7862 s** (**33.8067 tok/s**, **6.15×**), short TTFT **0.5710 s**. The table above stays the first sample.

## B5: paged bytes versus B4's exact-size cache

Not a timed run, not a load test, and not a percentile. B4 grows each sequence with `Tensor::cat` to the exact token count. It does not reserve a max length. The baseline is that exact size.

`paged_bytes_exceed_exact_cat_by_less_than_one_block_per_sequence` repeats this length list until another sequence would pass 256 blocks, which is 51 sequences:

`8, 16, 17, 24, 31, 32, 40, 48, 63, 64, 80, 96, 100, 127, 128, 160, 200, 255`

Qwen2.5-0.5B shapes: 24 layers, 2 kv heads, head dim 64, f32. One token of K and V is `24 * 2 * 2 * 64 * 4` = **24,576** bytes. The test checks that formula against `Tensor::elem_count` on one exact-length K tensor and against `PagedKvPool::kv_bytes` for a pool that holds exactly those sequences' blocks. One pass. No warmup.

| store | bytes for the same 51 sequences |
|---|---|
| B4 exact `Tensor::cat` | **94,666,752** |
| paged blocks | **99,090,432** |

Paged / exact = 99,090,432 / 94,666,752 = **1.0467**. Paged is larger by **4,423,680** bytes, which is **180** token slots, 180 / 51 = **3.53** slots each, under one block of 16. The sequence count is the same. The ≥2× packing item is **not met** against B4. What paging does provide is a fixed pool, waste bounded by one block per sequence, and preemption of the newest sequence when that pool is full.

A max-length reservation is a different design, not the B4 cache. For the record only: 8 sequences reserved at 512 tokens, or 16 reserved at 255, would occupy the 100,663,296 bytes of a 256-block pool, which is where a 6.375× or 3.1875× figure would come from. Those ratios are not B4, and they are not a result of this milestone.

Measured 2026-11-10, branch `feat/b5-paged-kv-cache`, rustc 1.99.1 (58cf8f9c3 2026-10-13), candle-core 0.11.1, `DType::F32`, `Device::Cpu`. Same machine class as the B4 row.

- Linux 6.12.113+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores.
- MemTotal 16,387,328 kB

## B6: open-loop offered load, 50 requests per cell

Not a steady-state run. Each cell is **one** sample: 1 warmup request (dropped) and **50** measured requests, seed 1, temperature 0. Every cell returned 50 ok, 0 errors, 1600 output tokens (32 per request). Output tokens/s is 1600 divided by the client wall clock, and that clock includes the Poisson gaps, so a low offered load is partly idle time. There is no closed-loop concurrency mode. The column is offered load, the Poisson `--rate` in requests/s. It is not a concurrency cap.

TTFT is the client clock from send to the first SSE `data:` line that has `delta.content`. The server writes a role `data:` line as soon as the stream is polled, before prefill returns. That line is not the first token. The client timeout is 900 s. The histogram records up to 30 minutes.

B2 has no HTTP server. "No batching" is the B5 `qwen-server` binary with `--max-batch-size 1`. The KV pool is still the B5 paged pool. It is not the original in-process `generate` loop. The B4 rows are a separate binary: the contiguous engine at main `3378a6b940ec1fef938962e5ec684df42ee5f3f8`, built in a detached worktree. That commit's CLI has no `--num-blocks`. The 2026-11-08 B4 row above (one request 5.64 tok/s, eight concurrent 34.38 tok/s, 16 tokens, a closed burst) is a different experiment and is not copied into this table.

The batch cap is fixed for the sweep. No-batching stays at 1. B4 and B5 stay at 8, the server default. `--max-waiting` is 64 so the 50 requests can sit in the queue instead of taking HTTP 429. B5 and no-batching also pass `--num-blocks 256`.

### Workload

`--workload mix`, seed 1. Two `StdRng` streams are both seeded with that integer and do not share draws. One draws the exponential gaps. The other draws `random_range(0..4)` once per measured request. The warmup is always the first catalog entry and is not part of that draw.

The user string is `"amber ".repeat(n)`, trailing space included. The server wraps it in the Qwen2.5 chat template: the default system line `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`, then the user string, then `<|im_start|>assistant\n`. `RankTokenizer::load_qwen` `encode` (every special token allowed) of that full string is the prompt length:

| repeats of `"amber "` | full prompt tokens | max_tokens |
|---|---|---|
| 2 | 32 | 32 |
| 34 | 64 | 32 |
| 66 | 96 | 32 |
| 98 | 128 | 32 |

Seed 1 over 50 requests assigned those four lengths **13, 14, 9, and 14** times. The first eight indexes are 3, 3, 3, 2, 0, 1, 0, 0. Every measured request produced 32 content events, so none of them stopped on EOS before `max_tokens`.

One request each, same machine, paged server, `--max-batch-size 1`, not a matrix cell: prompt 32/64/96/128 tokens had client TTFT 0.243/0.308/0.401/0.468 s and end-to-end 6.083/6.205/6.319/6.324 s. Decode was about 5.8–5.9 s. A weighted mean of those four TTFTs with the seed-1 counts is 0.35 s. That is the unloaded reference for the table below.

### Commands

Paged server and loadgen are the release build of `feat/b6-benchmarks` at `f410b6a73086778f3d589d1c99a2377d0020c743`.

```
cargo build --release -p qwen-server -p loadgen
target/release/qwen-server --bind 127.0.0.1:18080 --max-batch-size <1 or 8> --max-waiting 64 --num-blocks 256
target/release/loadgen --url http://127.0.0.1:18080/v1/chat/completions --requests 50 --rate <offered> --seed 1 --warmup 1 --workload mix
```

Contiguous B4 server. The current branch was not moved. `CARGO_TARGET_DIR` keeps that build out of this tree's `target/`. The build was run again for this round; the worktree was already at that commit and cargo reused the release binary.

```
git worktree add --detach /tmp/b4-wt 3378a6b940ec1fef938962e5ec684df42ee5f3f8
cd /tmp/b4-wt && CARGO_TARGET_DIR=/tmp/b4-target cargo build --release -p qwen-server
/tmp/b4-target/release/qwen-server --bind 127.0.0.1:18080 --max-batch-size 8 --max-waiting 64
```

Model: Qwen/Qwen2.5-0.5B-Instruct, f32, `Device::Cpu`. Measured 2026-11-10, rustc 1.99.1 (58cf8f9c3 2026-10-13).

- Linux 6.12.113+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores, one socket.
- MemTotal 16,387,328 kB

The 0.06 cells take about 14 minutes each because the arrivals themselves span `(50-1)/0.06` seconds. The first six cells used 3505 s. The four cells at 0.5 and 1.0 req/s are B4 and B5 only, same mix, seed, and batch cap, and finished in 375 s. 2.0 req/s was not run: 1.0 already saturates both of those modes. The batch-size-1 proxy has no row above 0.25 req/s, because that point is already saturated. 100 requests was not run.

| mode | offered load | max batch | n | output tok/s | req/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | e2e p99 |
|---|---|---|---|---|---|---|---|---|---|---|
| no batching | 0.06 req/s | 1 | 50 | 1.8782 | 0.0587 | 0.539 s | 14.819 s | 0.198 s | 0.248 s | 20.972 s |
| no batching | 0.25 req/s | 1 | 50 | 5.0282 | 0.1571 | 68.092 s | 110.952 s | 0.190 s | 0.222 s | 116.851 s |
| B4 contiguous | 0.06 req/s | 8 | 50 | 1.8960 | 0.0593 | 0.393 s | 0.733 s | 0.192 s | 0.328 s | 7.193 s |
| B4 contiguous | 0.25 req/s | 8 | 50 | 7.7556 | 0.2424 | 0.464 s | 1.083 s | 0.132 s | 0.664 s | 7.356 s |
| B4 contiguous | 0.5 req/s | 8 | 50 | 15.1531 | 0.4735 | 0.579 s | 4.135 s | 0.144 s | 0.785 s | 12.149 s |
| B4 contiguous | 1.0 req/s | 8 | 50 | 26.0757 | 0.8149 | 5.026 s | 9.781 s | 0.191 s | 0.903 s | 18.285 s |
| B5 paged | 0.06 req/s | 8 | 50 | 1.8962 | 0.0593 | 0.392 s | 1.594 s | 0.189 s | 0.316 s | 7.713 s |
| B5 paged | 0.25 req/s | 8 | 50 | 7.7725 | 0.2429 | 0.476 s | 1.223 s | 0.123 s | 0.730 s | 8.610 s |
| B5 paged | 0.5 req/s | 8 | 50 | 15.1789 | 0.4743 | 0.504 s | 2.738 s | 0.132 s | 0.708 s | 9.978 s |
| B5 paged | 1.0 req/s | 8 | 50 | 24.0145 | 0.7505 | 7.950 s | 12.255 s | 0.206 s | 1.148 s | 21.365 s |

Saturation here means the achieved request rate falls behind the offer, or TTFT p50 jumps. No batching does that at **0.25 req/s**: achieved 0.1571 against 0.25, and TTFT p50 goes from 0.539 s to 68.092 s. At 0.06 req/s it was still keeping up (0.0587 against 0.06). B4 and B5 do not do that at 0.5 req/s. Achieved rates are 0.4735 and 0.4743 against 0.5, TTFT p50 stays at 0.579 s and 0.504 s, and output rate is still climbing (7.76 to 15.15 tok/s for B4, 7.77 to 15.18 for B5). Both saturate at **1.0 req/s**. B4 achieves 0.8149 against 1.0 and TTFT p50 jumps to 5.026 s. B5 achieves 0.7505 against 1.0 and TTFT p50 jumps to 7.950 s. Nothing between 0.5 and 1.0 was measured, so the point named above is the first measured offer where that happened, not a fitted knee. In this load test, B5 paged is slightly behind B4 contiguous at 1.0 req/s (unprofiled), and the large gain comes from batching, not paging.

The tok/s gap at 0.06 (1.88 vs 1.90) is the idle time, not the engine. CSV: `bench/results/b6-ttft-curve.csv`. Each `#` is about 1 s of p99 TTFT. The saturated no-batching cell and the 1.0 req/s p99s are past a 16 s scale, so those p99s are written as numbers.

```
offered 0.06 req/s
no-batch  1.88 tok/s  ###############  14.8 s
B4        1.90 tok/s  #                  0.7 s
B5        1.90 tok/s  ##                 1.6 s

offered 0.25 req/s
no-batch  5.03 tok/s  p99 TTFT 111.0 s
B4        7.76 tok/s  #                  1.1 s
B5        7.77 tok/s  #                  1.2 s

offered 0.5 req/s
B4       15.15 tok/s  ####               4.1 s
B5       15.18 tok/s  ###                2.7 s

offered 1.0 req/s
B4       26.08 tok/s  p99 TTFT 9.8 s   p50 TTFT 5.0 s
B5       24.01 tok/s  p99 TTFT 12.3 s  p50 TTFT 8.0 s
```

## Prefix cache: one shared system prefix, sharing on and off

Not a sweep and not a steady-state run. One sample each. Warmup 1 (dropped from the histograms; it does fill the prefix cache when sharing is on) and **8** measured requests, seed 1, offered load 2 req/s, `max_tokens` 8, temperature 0, `--max-batch-size` 4, `--num-blocks` 256. Both cells returned 8 ok, 0 errors, and 16 output tokens. The model stopped on EOS before `max_tokens`, so this is not an 8-token generation. p99 of 8 points is the slow top of that sample. Output tok/s includes the Poisson gaps.

The system string is `"amber "` repeated 90 times (91 tokens). The chat template puts 99 tokens on the wire before the user suffix, so the first six blocks (96 tokens) are full and shared. The user suffix is one of `alpha`, `bravo`, `charlie`, `delta`, drawn by `prefix_suffixes(1, 8)`, which is indexes 3, 3, 3, 2, 0, 1, 0, 0. The partial block that contains the suffix is private. Sharing off is the same server with `--no-prefix-cache`.

```
cargo build --release -p qwen-server -p loadgen
target/release/qwen-server --bind 127.0.0.1:18080 --max-batch-size 4 --max-waiting 16 --num-blocks 256
target/release/qwen-server --bind 127.0.0.1:18080 --max-batch-size 4 --max-waiting 16 --num-blocks 256 --no-prefix-cache
target/release/loadgen --url http://127.0.0.1:18080/v1/chat/completions --requests 8 --rate 2 --seed 1 --warmup 1 --workload prefix
```

Model: Qwen/Qwen2.5-0.5B-Instruct, f32, `Device::Cpu`. Measured 2026-11-11, rustc 1.99.1 (58cf8f9c3 2026-10-13). Same machine as the B6 rows.

- Linux 6.12.113+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports "Intel(R) Xeon(R) Processor"). One thread per core, four cores, one socket.
- MemTotal 16,387,328 kB

`peak_live` is the most blocks that had a positive refcount at once, printed when the process stopped. Cached blocks at refcount 0 are not in that count.

| sharing | n | output tok/s | req/s | TTFT p50 | TTFT p99 | ITL p50 | ITL p99 | e2e p99 | peak live blocks |
|---|---|---|---|---|---|---|---|---|---|
| on | 8 | 3.6527 | 1.8263 | 0.144 s | 0.445 s | 0.183 s | 0.511 s | 1.632 s | 9 |
| off | 8 | 2.9230 | 1.4615 | 0.481 s | 1.325 s | 0.186 s | 1.108 s | 3.449 s | 28 |

With sharing on, TTFT p50 is 0.144 s against 0.481 s off, and the pool's peak live blocks are 9 against 28. One sample of 8 requests. The warmup request is why the measured rows already hit the cache.
