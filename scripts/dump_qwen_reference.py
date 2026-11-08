#!/usr/bin/env python3
"""Dump float32 prompt logits and 20 greedy tokens for Qwen2.5-0.5B-Instruct.

The Rust ignored test compares `QwenModel::last_logits` to the f32 file and
`greedy_tokens` to the ids in the json. Both sides are float32. The
checkpoint on disk is bf16; this process casts on load, the same cast candle
does with `DType::F32`.

Greedy is argmax, one token at a time, with the whole sequence recomputed.
Generation does not stop on eos and does not sample. `generation_config.json`
has `do_sample: true`; that file is not used here.

Usage:
    scripts/dump_qwen_reference.py [model_dir]

Default model dir:
    ~/.cache/tokenizer/qwen2.5-0.5b-instruct
"""

import json
import sys
from pathlib import Path

import torch
import transformers
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM

EXPECTED_TORCH = "2.15.0+cpu"
EXPECTED_TRANSFORMERS = "5.22.1"
EXPECTED_TOKENIZERS = "0.23.3"

if torch.__version__ != EXPECTED_TORCH:
    sys.exit(f"torch {torch.__version__} != {EXPECTED_TORCH}")
if transformers.__version__ != EXPECTED_TRANSFORMERS:
    sys.exit(f"transformers {transformers.__version__} != {EXPECTED_TRANSFORMERS}")
import tokenizers

if tokenizers.__version__ != EXPECTED_TOKENIZERS:
    sys.exit(f"tokenizers {tokenizers.__version__} != {EXPECTED_TOKENIZERS}")

PROMPT = "The capital of France is"
NEW_TOKENS = 20
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = Path.home() / ".cache/tokenizer/qwen2.5-0.5b-instruct"


def main():
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DIR
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    prompt_ids = tok.encode(PROMPT).ids
    bare = tok.encode(PROMPT, add_special_tokens=False).ids
    if prompt_ids != bare:
        sys.exit(f"add_special_tokens changed the prompt ids: {prompt_ids} vs {bare}")

    torch.set_num_threads(4)
    # `dtype` is the transformers 5.22 name. `torch_dtype` still works and
    # warns. The checked-in fixture was produced with that alias; both pass
    # float32 through to the same loader.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    ids = list(prompt_ids)
    generated = []
    prompt_logits = None
    with torch.no_grad():
        for step in range(NEW_TOKENS):
            logits = model(torch.tensor([ids], dtype=torch.long)).logits[0, -1]
            logits = logits.float().cpu()
            if step == 0:
                prompt_logits = logits.numpy().astype("<f4", copy=False)
            nxt = int(torch.argmax(logits).item())
            generated.append(nxt)
            ids.append(nxt)

    if prompt_logits is None:
        sys.exit("no logits")
    out_dir = ROOT / "tests/fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)
    f32_path = out_dir / "qwen_last_logits.f32"
    prompt_logits.tofile(f32_path)

    topk = torch.topk(torch.from_numpy(prompt_logits.copy()), k=16)
    meta = {
        "prompt": PROMPT,
        "prompt_ids": prompt_ids,
        "greedy_ids": generated,
        "greedy_text": tok.decode(generated),
        "vocab": int(prompt_logits.shape[0]),
        "argmax": int(generated[0]),
        "topk": [
            {"id": int(i), "logit": float(v)}
            for i, v in zip(topk.indices.tolist(), topk.values.tolist())
        ],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "dtype": "float32",
        "attn_implementation": "eager",
        "new_tokens": NEW_TOKENS,
    }
    json_path = out_dir / "qwen_greedy.json"
    json_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"prompt ids: {prompt_ids}")
    print(f"greedy ids: {generated}")
    print(f"greedy text: {meta['greedy_text']!r}")
    print(f"argmax {meta['argmax']} logit {meta['topk'][0]['logit']}")
    print(f"wrote {f32_path} ({f32_path.stat().st_size} bytes) and {json_path}")


if __name__ == "__main__":
    main()
