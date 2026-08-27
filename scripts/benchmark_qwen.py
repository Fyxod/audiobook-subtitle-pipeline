#!/usr/bin/env python3
"""Compare center-channel and conservative-downmix ASR on a short sample."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from qwen_asr import Qwen3ASRModel


CONTEXT = """Transcribe this long-form English audio faithfully and verbatim.
Preserve sentence punctuation and capitalization. Do not describe music or sound effects."""


def serialise(result: object) -> dict:
    stamps = getattr(result, "time_stamps", None)
    return {
        "language": getattr(result, "language", None),
        "text": getattr(result, "text", ""),
        "words": [
            {
                "text": item.text,
                "start": item.start_time,
                "end": item.end_time,
            }
            for item in (getattr(stamps, "items", None) or [])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument(
        "--context-file",
        type=Path,
        help="optional UTF-8 file containing source-specific names or domain terms",
    )
    args = parser.parse_args()
    paths = args.audio
    context = CONTEXT
    if args.context_file:
        extra = args.context_file.read_text(encoding="utf-8").strip()
        if extra:
            context += f"\nReference spellings and domain terms:\n{extra}"

    started = time.time()
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_inference_batch_size=2,
        max_new_tokens=4096,
        forced_aligner_kwargs={
            "device_map": "cuda:0",
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )
    loaded = time.time()
    results = model.transcribe(
        audio=[str(path) for path in paths],
        context=[context] * len(paths),
        language=["English"] * len(paths),
        return_time_stamps=True,
    )
    finished = time.time()
    payload = {
        "model": "Qwen/Qwen3-ASR-1.7B",
        "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        "load_seconds": round(loaded - started, 3),
        "inference_seconds": round(finished - loaded, 3),
        "results": {str(path): serialise(result) for path, result in zip(paths, results)},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
