#!/usr/bin/env python3
"""Checkpointed Whisper Large-v3 audit transcription."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-group-size", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    chunks = manifest["chunks"]
    audio_path = Path(manifest["audio"])
    done = load_done(args.output)
    pending = [chunk for chunk in chunks if chunk["id"] not in done]
    print(f"Whisper: {len(done)} complete, {len(pending)} pending", flush=True)
    if not pending:
        return 0

    model_id = "openai/whisper-large-v3"
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
        stride_length_s=(5, 5),
        batch_size=args.batch_size,
        return_timestamps="word",
        ignore_warning=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(audio_path) as audio, args.output.open("a", encoding="utf-8") as output:
        for group_start in range(0, len(pending), args.input_group_size):
            group = pending[group_start : group_start + args.input_group_size]
            inputs = []
            for chunk in group:
                audio.seek(chunk["start_frame"])
                samples = audio.read(chunk["end_frame"] - chunk["start_frame"], dtype="float32")
                inputs.append({"raw": samples, "sampling_rate": audio.samplerate})
            started = time.time()
            results = asr(
                inputs,
                generate_kwargs={"language": "english", "task": "transcribe"},
            )
            elapsed = (time.time() - started) / len(group)
            for chunk, result in zip(group, results):
                offset = float(chunk["start"])
                words = []
                for item in result.get("chunks", []):
                    start, end = item.get("timestamp", (None, None))
                    if start is None:
                        continue
                    if end is None:
                        end = chunk["duration"]
                    words.append(
                        {
                            "text": item.get("text", "").strip(),
                            "start": round(offset + float(start), 3),
                            "end": round(offset + float(end), 3),
                        }
                    )
                record = {
                    **chunk,
                    "model": model_id,
                    "language": "English",
                    "text": result.get("text", "").strip(),
                    "words": words,
                    "elapsed_seconds": round(elapsed, 3),
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                done.add(chunk["id"])
                print(
                    f"[{len(done):03d}/{len(chunks):03d}] {chunk['id']} "
                    f"{len(words)} words in {elapsed:.1f}s",
                    flush=True,
                )
            del inputs, results
            gc.collect()
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
