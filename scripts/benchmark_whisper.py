#!/usr/bin/env python3
"""Run Whisper Large-v3 on one or more benchmark samples."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def main() -> int:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: benchmark_whisper.py AUDIO [AUDIO ...]")

    model_id = "openai/whisper-large-v3"
    started = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="sdpa",
    ).to("cuda:0")
    processor = AutoProcessor.from_pretrained(model_id)
    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.bfloat16,
        device="cuda:0",
        chunk_length_s=30,
        batch_size=4,
        return_timestamps="word",
    )
    loaded = time.time()
    results = asr(
        [str(path) for path in paths],
        generate_kwargs={"language": "english", "task": "transcribe"},
    )
    finished = time.time()
    if isinstance(results, dict):
        results = [results]
    payload = {
        "model": model_id,
        "load_seconds": round(loaded - started, 3),
        "inference_seconds": round(finished - loaded, 3),
        "results": {str(path): result for path, result in zip(paths, results)},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
