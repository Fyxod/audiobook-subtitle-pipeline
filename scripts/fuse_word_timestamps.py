#!/usr/bin/env python3
"""Fuse independent word timings without importing bad cross-chunk matches.

Qwen's forced aligner supplies the authoritative word sequence and final text.
Whisper is used only as an independent timing anchor when a locally consistent
word match agrees within a plausible offset. Large discrepancies are rejected:
they usually mean that the two ASRs inserted or omitted dialogue (common in a
multi-speaker recording), and copying those timestamps would create late subtitles.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path


def load_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        return {
            record["id"]: record
            for line in handle
            if line.strip()
            for record in [json.loads(line)]
        }


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def index_map(source: list[dict], target: list[dict]) -> dict[int, int]:
    a = [norm(item["text"]) for item in source]
    b = [norm(item["text"]) for item in target]
    mapping: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            mapping.update(zip(range(i1, i2), range(j1, j2)))
    return mapping


def local_consistent(candidates: list[tuple[int, int, float]], index: int) -> bool:
    """Accept an anchor only when nearby matched words share its offset."""
    delta = candidates[index][2]
    nearby = [
        item[2]
        for item in candidates[max(0, index - 4) : index + 5]
        if abs(item[2]) <= 4.0
    ]
    if not nearby:
        return False
    nearby.sort()
    median = nearby[len(nearby) // 2]
    return abs(delta) <= 4.0 and abs(delta - median) <= 0.8


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("aligned", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("whisper", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    aligned = load_jsonl(args.aligned)
    qwen = load_jsonl(args.qwen)
    whisper = load_jsonl(args.whisper)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = fused = rejected = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record_id, record in aligned.items():
            a_words = record["words"]
            q_words = qwen[record_id]["words"]
            w_words = whisper[record_id]["words"]
            a_to_q = index_map(a_words, q_words)
            q_to_w = index_map(q_words, w_words)
            candidates: list[tuple[int, int, float]] = []
            for a_index, q_index in a_to_q.items():
                if q_index in q_to_w:
                    w_index = q_to_w[q_index]
                    candidates.append(
                        (a_index, w_index, w_words[w_index]["start"] - q_words[q_index]["start"])
                    )
            candidates.sort()
            by_a = {item[0]: item for item in candidates}
            out_words = []
            previous_start = float("-inf")
            record_fused = 0
            for a_index, word in enumerate(a_words):
                total += 1
                candidate = by_a.get(a_index)
                if candidate is not None:
                    _, w_index, delta = candidate
                    candidate_index = candidates.index(candidate)
                    if local_consistent(candidates, candidate_index):
                        q_index = a_to_q[a_index]
                        w_word = w_words[w_index]
                        # Use the earlier onset and later offset only when the
                        # two timestamps are locally compatible. This directly
                        # addresses occasional late cue starts while preserving
                        # Qwen's timing when Whisper has a bad insertion match.
                        word = {
                            **word,
                            "start": round(
                                max(previous_start, min(word["start"], w_word["start"])), 3
                            ),
                            "end": round(max(word["end"], w_word["end"]), 3),
                        }
                        fused += 1
                        record_fused += 1
                    else:
                        rejected += 1
                word["start"] = round(max(previous_start, word["start"]), 3)
                word["end"] = round(max(word["end"], word["start"]), 3)
                previous_start = word["start"]
                out_words.append(word)
            # Whisper can occasionally attach a word to a long overlapping
            # segment, producing an end time many seconds after the word was
            # spoken. Prevent that from holding a subtitle on screen or
            # reordering the next cue. The next word is the strongest local
            # upper bound; 1.5 s is a conservative ceiling for one word.
            for word_index, word in enumerate(out_words):
                next_start = (
                    out_words[word_index + 1]["start"]
                    if word_index + 1 < len(out_words)
                    else word["start"] + 1.5
                )
                upper = min(word["start"] + 1.5, next_start) if next_start > word["start"] else word["start"] + 1.5
                word["end"] = round(min(max(word["end"], word["start"]), upper), 3)
            handle.write(
                json.dumps(
                    {**record, "words": out_words, "fused_word_timings": record_fused},
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(json.dumps({"words": total, "fused": fused, "rejected_matches": rejected}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
