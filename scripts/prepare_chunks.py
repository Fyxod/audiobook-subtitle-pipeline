#!/usr/bin/env python3
"""Create chapter-aware, low-energy ASR chunk boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def quiet_boundary(
    audio: sf.SoundFile,
    target_frame: int,
    search_frames: int,
    lower_frame: int,
    upper_frame: int,
) -> int:
    start = max(lower_frame, target_frame - search_frames)
    stop = min(upper_frame, target_frame + search_frames)
    audio.seek(start)
    samples = audio.read(stop - start, dtype="float32", always_2d=False)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    window = max(1, round(audio.samplerate * 0.05))
    count = len(samples) // window
    if count < 2:
        return target_frame
    framed = samples[: count * window].reshape(count, window)
    rms = np.sqrt(np.mean(framed * framed, axis=1) + 1e-12)

    # Choose the point nearest the target among the quietest 8% of windows.
    threshold = np.quantile(rms, 0.08)
    candidates = np.flatnonzero(rms <= threshold)
    target_index = (target_frame - start) / window
    chosen = int(candidates[np.argmin(np.abs(candidates - target_index))])
    return start + chosen * window + window // 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("ffprobe_json", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-seconds", type=float, default=210.0)
    parser.add_argument("--search-seconds", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=270.0)
    args = parser.parse_args()

    probe = json.loads(args.ffprobe_json.read_text(encoding="utf-8"))
    chapters = probe["chapters"]
    chunks: list[dict] = []

    with sf.SoundFile(args.audio) as audio:
        sr = audio.samplerate
        if sr != 16000 or audio.channels != 1:
            raise SystemExit(f"expected mono 16 kHz WAV, got {audio.channels}ch @ {sr} Hz")

        target = round(args.target_seconds * sr)
        search = round(args.search_seconds * sr)
        maximum = round(args.max_seconds * sr)

        for chapter in chapters:
            chapter_start = round(float(chapter["start_time"]) * sr)
            chapter_end = min(round(float(chapter["end_time"]) * sr), len(audio))
            cursor = chapter_start
            part = 1
            while cursor < chapter_end:
                remaining = chapter_end - cursor
                if remaining <= maximum:
                    boundary = chapter_end
                else:
                    wanted = cursor + target
                    boundary = quiet_boundary(
                        audio,
                        target_frame=wanted,
                        search_frames=search,
                        lower_frame=cursor + round(120 * sr),
                        upper_frame=min(cursor + maximum, chapter_end - round(60 * sr)),
                    )
                chunks.append(
                    {
                        "id": f"c{int(chapter['id']):02d}_p{part:02d}",
                        "chapter_id": int(chapter["id"]),
                        "chapter_title": chapter.get("tags", {}).get("title", f"Chapter {chapter['id']}"),
                        "part": part,
                        "start_frame": cursor,
                        "end_frame": boundary,
                        "start": round(cursor / sr, 3),
                        "end": round(boundary / sr, 3),
                        "duration": round((boundary - cursor) / sr, 3),
                    }
                )
                cursor = boundary
                part += 1

    manifest = {
        "audio": str(args.audio),
        "sample_rate": 16000,
        "target_seconds": args.target_seconds,
        "search_seconds": args.search_seconds,
        "max_seconds": args.max_seconds,
        "chunks": chunks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {len(chunks)} chunks to {args.output}")
    print(f"duration range: {min(c['duration'] for c in chunks):.3f}s - {max(c['duration'] for c in chunks):.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
