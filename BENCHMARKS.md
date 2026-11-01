# Benchmarks

## A1: naive training baseline

Measured 2026-11-01 on commit `15f5bc2b00ec2fe25cab90ac24788f31f8a2f6ff`.

Command (hyperfine 2.0.0, 2 warmups, then 10 runs):

```
hyperfine --warmup 2 'target/release/tokenizer book.txt "hello"'
```

Wall time: **4.206 s ± 0.036 s** (mean 4.206191628 s, σ 0.036481657 s). Min 4.171 s, max 4.273 s.

Machine:

- Linux 6.12.94+ x86_64, KVM guest
- 4 vCPUs, Intel Xeon (family 6, model 207; `lscpu` reports the generic name "Intel(R) Xeon(R) Processor")
- MemTotal 16,387,280 kB
- rustc 1.99.1 (58cf8f9c3 2026-10-13)

Corpus: Pride and Prejudice, Project Gutenberg #1342, saved locally as `book.txt` (gitignored) from <https://www.gutenberg.org/cache/epub/1342/pg1342.txt>. Training learned all 1,000 requested merges.

Compression: **3.239 bytes/token** (772,386 bytes / 238,459 tokens). The token count is the length of the training sequence after those 1,000 merges, which is the book under the learned rules. The timed command also encodes the probe `"hello"` (5 bytes → 3 tokens); that probe is not the ratio above.
