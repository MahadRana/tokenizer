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
