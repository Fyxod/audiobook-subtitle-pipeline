#!/usr/bin/env python3
"""Benchmark the CTranslate2 Whisper Large-v3 implementation."""

from __future__ import annotations

import json
import sys
import time

from faster_whisper import WhisperModel


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: benchmark_faster_whisper.py AUDIO")
    started = time.time()
    model = WhisperModel(
        "models/faster-whisper/large-v3",
        device="cuda",
        compute_type="float16",
    )
    loaded = time.time()
    segments, info = model.transcribe(
        sys.argv[1],
        language="en",
        beam_size=5,
        best_of=5,
        temperature=0,
        condition_on_previous_text=True,
        word_timestamps=True,
        vad_filter=False,
    )
    segments = list(segments)
    finished = time.time()
    payload = {
        "model": "large-v3",
        "load_seconds": round(loaded - started, 3),
        "inference_seconds": round(finished - loaded, 3),
        "language": info.language,
        "text": "".join(segment.text for segment in segments).strip(),
        "words": [
            {"text": word.word.strip(), "start": word.start, "end": word.end}
            for segment in segments
            for word in (segment.words or [])
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
