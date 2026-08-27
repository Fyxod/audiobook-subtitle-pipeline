#!/usr/bin/env python3
"""Force-align the reconciled transcript and restore its surface punctuation."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import soundfile as sf
import torch
from qwen_asr import Qwen3ForcedAligner


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {record["id"]: record for line in handle if line.strip() for record in [json.loads(line)]}


def chars(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def restore_surface(text: str, raw_items: list[dict]) -> tuple[list[dict], bool]:
    surfaces = text.split()
    surface_norm = [chars(token) for token in surfaces]
    item_norm = [chars(item["text"]) for item in raw_items]
    text_chars = "".join(surface_norm)
    item_chars = "".join(item_norm)

    if not text_chars or text_chars != item_chars:
        return raw_items, False

    surface_ranges = []
    cursor = 0
    for token in surface_norm:
        surface_ranges.append((cursor, cursor + len(token)))
        cursor += len(token)

    item_ranges = []
    cursor = 0
    for token in item_norm:
        item_ranges.append((cursor, cursor + len(token)))
        cursor += len(token)

    restored = []
    item_index = 0
    for surface, (surface_start, surface_end) in zip(surfaces, surface_ranges):
        while item_index < len(item_ranges) and item_ranges[item_index][1] <= surface_start:
            item_index += 1
        covered = []
        scan = item_index
        while scan < len(item_ranges) and item_ranges[scan][0] < surface_end:
            if item_ranges[scan][1] > surface_start:
                covered.append(raw_items[scan])
            scan += 1
        if not covered:
            return raw_items, False
        restored.append(
            {
                "text": surface,
                "start": covered[0]["start"],
                "end": covered[-1]["end"],
            }
        )
    return restored, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("consensus", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    consensus = load_jsonl(args.consensus)
    existing = load_jsonl(args.output)
    pending = [c for c in manifest["chunks"] if c["id"] not in existing]
    print(f"Alignment: {len(existing)} complete, {len(pending)} pending", flush=True)
    if not pending:
        return 0

    aligner = Qwen3ForcedAligner.from_pretrained(
        "Qwen/Qwen3-ForcedAligner-0.6B",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    audio_path = Path(manifest["audio"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with sf.SoundFile(audio_path) as audio, args.output.open("a", encoding="utf-8") as output:
        for batch_start in range(0, len(pending), args.batch_size):
            batch = pending[batch_start : batch_start + args.batch_size]
            waveforms = []
            texts = []
            for chunk in batch:
                audio.seek(chunk["start_frame"])
                samples = audio.read(chunk["end_frame"] - chunk["start_frame"], dtype="float32")
                waveforms.append((samples, audio.samplerate))
                texts.append(consensus[chunk["id"]]["text"])

            started = time.time()
            results = aligner.align(waveforms, texts, ["English"] * len(batch))
            per_item = (time.time() - started) / len(batch)
            for chunk, text, result in zip(batch, texts, results):
                offset = float(chunk["start"])
                raw_words = [
                    {
                        "text": item.text,
                        "start": round(offset + item.start_time, 3),
                        "end": round(offset + item.end_time, 3),
                    }
                    for item in result.items
                ]
                words, restored = restore_surface(text, raw_words)
                record = {
                    **chunk,
                    "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
                    "text": text,
                    "words": words,
                    "raw_word_count": len(raw_words),
                    "surface_word_count": len(words),
                    "surface_restored": restored,
                    "elapsed_seconds": round(per_item, 3),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                existing[chunk["id"]] = record
                print(
                    f"[{len(existing):03d}/{len(manifest['chunks']):03d}] {chunk['id']} "
                    f"{len(words)} words surface={restored} in {per_item:.1f}s",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
