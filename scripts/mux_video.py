#!/usr/bin/env python3
"""Mux original audio, chapters, black video, and selectable subtitles."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size:stream=index,codec_name,codec_type,channels,channel_layout:stream_tags=language,title:stream_disposition=default:chapter=id,start_time,end_time:chapter_tags=title",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("subtitles", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source_probe = probe(args.audio)
    duration = float(source_probe["format"]["duration"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "warning",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=1280x720:r=1",
        "-i",
        str(args.audio),
        "-i",
        str(args.subtitles),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:0",
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "stillimage",
        "-crf",
        "40",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-c:s",
        "srt",
        "-metadata:s:v:0",
        "title=Black screen",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:0",
        "title=Original audio",
        "-metadata:s:s:0",
        "language=eng",
        "-metadata:s:s:0",
        "title=English (AI-generated; Qwen + Whisper)",
        "-disposition:s:0",
        "default",
        "-t",
        f"{duration:.3f}",
        str(args.output),
    ]
    subprocess.run(command, check=True)
    output_probe = probe(args.output)
    verification = {
        "source_duration": duration,
        "output_duration": float(output_probe["format"]["duration"]),
        "output_size": int(output_probe["format"]["size"]),
        "streams": output_probe.get("streams", []),
        "chapter_count": len(output_probe.get("chapters", [])),
    }
    verification_path = args.output.with_suffix(".verification.json")
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
