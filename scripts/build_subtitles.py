#!/usr/bin/env python3
"""Build readable SRT/VTT cues, transcript, word timings, and QC report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MAX_LINE = 42
MAX_CUE_CHARS = 78
MAX_CUE_SECONDS = 6.5


def load_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {record["id"]: record for line in handle if line.strip() for record in [json.loads(line)]}


def join_words(words: list[dict]) -> str:
    return " ".join(word["text"] for word in words).strip()


def can_wrap_two_lines(text: str) -> bool:
    if len(text) <= MAX_LINE:
        return True
    tokens = text.split()
    return any(
        len(" ".join(tokens[:index])) <= MAX_LINE
        and len(" ".join(tokens[index:])) <= MAX_LINE
        for index in range(1, len(tokens))
    )


def should_break(current: list[dict], word: dict) -> bool:
    if not current:
        return False
    text = join_words(current)
    candidate = f"{text} {word['text']}"
    duration = current[-1]["end"] - current[0]["start"]
    candidate_duration = word["end"] - current[0]["start"]
    gap = word["start"] - current[-1]["end"]
    if gap >= 1.0:
        return True
    if len(candidate) > MAX_CUE_CHARS:
        return True
    if len(candidate) > MAX_LINE and not can_wrap_two_lines(candidate):
        return True
    if candidate_duration > MAX_CUE_SECONDS and duration >= 1.0:
        return True
    if re.search(r"[.!?][\"'’”)]?$", current[-1]["text"]) and duration >= 1.0 and len(text) >= 20:
        return True
    if re.search(r"[,;:][\"'’”)]?$", current[-1]["text"]) and duration >= 3.5 and len(text) >= 48:
        return True
    return False


def wrap_balanced(text: str) -> str:
    if len(text) <= MAX_LINE:
        return text
    tokens = text.split()
    candidates = []
    for index in range(1, len(tokens)):
        left = " ".join(tokens[:index])
        right = " ".join(tokens[index:])
        if len(left) <= MAX_LINE and len(right) <= MAX_LINE:
            score = max(len(left), len(right)) + abs(len(left) - len(right)) * 0.2
            candidates.append((score, left, right))
    if candidates:
        _, left, right = min(candidates)
        return f"{left}\n{right}"
    # A pathological long token is kept intact rather than silently dropping text.
    return text


def timestamp(seconds: float, comma: bool = True) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    separator = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("aligned", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--basename", default="audiobook")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.basename):
        parser.error("--basename must contain only letters, numbers, dots, underscores, or hyphens")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = load_jsonl(args.aligned)
    missing = [chunk["id"] for chunk in manifest["chunks"] if chunk["id"] not in records]
    if missing:
        raise SystemExit(f"missing {len(missing)} aligned chunks; first: {missing[0]}")

    ordered = [records[chunk["id"]] for chunk in manifest["chunks"]]
    words = [word for record in ordered for word in record["words"] if word["text"].strip()]
    words.sort(key=lambda item: (item["start"], item["end"]))

    cue_words: list[list[dict]] = []
    current: list[dict] = []
    for word in words:
        if should_break(current, word):
            cue_words.append(current)
            current = []
        current.append(word)
    if current:
        cue_words.append(current)

    cues = []
    for index, group in enumerate(cue_words):
        start = float(group[0]["start"])
        raw_end = max(float(word["end"]) for word in group)
        next_start = float(cue_words[index + 1][0]["start"]) if index + 1 < len(cue_words) else raw_end + 1.0
        desired_end = min(max(raw_end + 0.08, start + 0.8), start + MAX_CUE_SECONDS)
        end = (
            min(desired_end, next_start - 0.04)
            if next_start > start + 0.08
            else min(raw_end, start + MAX_CUE_SECONDS)
        )
        end = max(end, start + 0.08)
        text = wrap_balanced(join_words(group))
        cues.append({"index": index + 1, "start": start, "end": end, "text": text})

    # Enforce strictly non-overlapping cues after duration extension.
    for index in range(len(cues) - 1):
        if cues[index]["end"] >= cues[index + 1]["start"]:
            if cues[index + 1]["start"] <= cues[index]["start"]:
                # A few short words can share the same quantized aligner
                # timestamp. Keep the cues selectable and strictly ordered;
                # their following cue carries the remaining spoken text.
                cues[index]["end"] = min(
                    cues[index]["start"] + MAX_CUE_SECONDS,
                    max(cues[index]["end"], cues[index]["start"] + 0.8),
                )
                cues[index + 1]["start"] = cues[index]["end"] + 0.04
            else:
                cues[index]["end"] = max(
                    cues[index]["start"] + 0.01,
                    cues[index + 1]["start"] - 0.04,
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    srt_path = args.output_dir / f"{args.basename}.en.srt"
    with srt_path.open("w", encoding="utf-8", newline="\n") as handle:
        for cue in cues:
            handle.write(
                f"{cue['index']}\n{timestamp(cue['start'])} --> {timestamp(cue['end'])}\n"
                f"{cue['text']}\n\n"
            )

    vtt_path = args.output_dir / f"{args.basename}.en.vtt"
    with vtt_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("WEBVTT\n\n")
        for cue in cues:
            handle.write(
                f"{timestamp(cue['start'], comma=False)} --> {timestamp(cue['end'], comma=False)}\n"
                f"{cue['text']}\n\n"
            )

    transcript_path = args.output_dir / f"{args.basename}.transcript.txt"
    with transcript_path.open("w", encoding="utf-8", newline="\n") as handle:
        previous_chapter = None
        for record in ordered:
            if record["chapter_id"] != previous_chapter:
                if previous_chapter is not None:
                    handle.write("\n\n")
                handle.write(record["chapter_title"] + "\n\n")
                previous_chapter = record["chapter_id"]
            handle.write(record["text"].strip() + " ")
        handle.write("\n")

    word_path = args.output_dir / f"{args.basename}.word_timestamps.json"
    word_path.write_text(json.dumps(words, ensure_ascii=False, indent=2), encoding="utf-8")

    line_lengths = [len(line) for cue in cues for line in cue["text"].splitlines()]
    cps = [
        len(cue["text"].replace("\n", "")) / max(cue["end"] - cue["start"], 0.001)
        for cue in cues
    ]
    qc = {
        "chunks": len(ordered),
        "words": len(words),
        "cues": len(cues),
        "first_cue_start": cues[0]["start"] if cues else None,
        "last_cue_end": cues[-1]["end"] if cues else None,
        "max_line_length": max(line_lengths, default=0),
        "max_cue_duration": max((cue["end"] - cue["start"] for cue in cues), default=0),
        "overlap_count": sum(cues[i]["end"] >= cues[i + 1]["start"] for i in range(len(cues) - 1)),
        "cues_over_25_cps": sum(value > 25 for value in cps),
        "max_cps": round(max(cps, default=0), 3),
        "surface_restore_failures": [record["id"] for record in ordered if not record["surface_restored"]],
        "zero_duration_words_from_aligner_quantization": sum(
            word["end"] == word["start"] for word in words
        ),
        "negative_word_timings": sum(word["end"] < word["start"] for word in words),
    }
    (args.output_dir / "subtitle_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    print(json.dumps(qc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
