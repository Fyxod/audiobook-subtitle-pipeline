#!/usr/bin/env python3
"""Checkpointed Qwen3-ASR + Qwen3-ForcedAligner transcription."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_asr import Qwen3ASRModel


BASE_CONTEXT = """Faithfully transcribe every spoken English word in this long-form audio.
Do not summarize, modernize, censor, or describe music and sound effects. Preserve natural
sentence punctuation and capitalization."""


def transcription_context(context_file: Path | None) -> str:
    if context_file is None:
        return BASE_CONTEXT
    extra = context_file.read_text(encoding="utf-8").strip()
    return f"{BASE_CONTEXT}\nReference spellings and domain terms:\n{extra}" if extra else BASE_CONTEXT


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def serialise(chunk: dict, result: object, elapsed: float) -> dict:
    stamps = getattr(result, "time_stamps", None)
    offset = float(chunk["start"])
    return {
        **chunk,
        "model": "Qwen/Qwen3-ASR-1.7B",
        "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        "language": getattr(result, "language", None),
        "text": getattr(result, "text", ""),
        "words": [
            {
                "text": item.text,
                "start": round(offset + item.start_time, 3),
                "end": round(offset + item.end_time, 3),
            }
            for item in (getattr(stamps, "items", None) or [])
        ],
        "elapsed_seconds": round(elapsed, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--context-file",
        type=Path,
        help="optional UTF-8 file containing source-specific names or domain terms",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    chunks = manifest["chunks"]
    audio_path = Path(manifest["audio"])
    done = load_done(args.output)
    pending = [chunk for chunk in chunks if chunk["id"] not in done]
    context = transcription_context(args.context_file)
    print(f"Qwen: {len(done)} complete, {len(pending)} pending", flush=True)
    if not pending:
        return 0

    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B",
        forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        max_inference_batch_size=args.batch_size,
        max_new_tokens=4096,
        forced_aligner_kwargs={
            "device_map": "cuda:0",
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(audio_path) as audio, args.output.open("a", encoding="utf-8") as output:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            waveforms = []
            contexts = []
            for chunk in batch:
                audio.seek(chunk["start_frame"])
                waveforms.append(
                    (audio.read(chunk["end_frame"] - chunk["start_frame"], dtype="float32"), audio.samplerate)
                )
                contexts.append(
                    f"{context}\nSection: {chunk['chapter_title']}, part {chunk['part']}."
                )

            started = time.time()
            results = model.transcribe(
                audio=waveforms,
                context=contexts,
                language=["English"] * len(batch),
                return_time_stamps=True,
            )
            per_item = (time.time() - started) / len(batch)
            for chunk, result in zip(batch, results):
                record = serialise(chunk, result, per_item)
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                print(
                    f"[{len(done) + 1:03d}/{len(chunks):03d}] {chunk['id']} "
                    f"{len(record['words'])} words in {per_item:.1f}s",
                    flush=True,
                )
                done.add(chunk["id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
