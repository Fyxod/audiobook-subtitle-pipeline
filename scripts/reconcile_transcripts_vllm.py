#!/usr/bin/env python3
"""Reconcile ASR transcripts through a remote OpenAI-compatible vLLM server.

The remote Nemotron endpoint is used only for text adjudication. Audio and all
timing work remain local, so this can run concurrently with the next book's
local ASR job. Nemotron emits a hidden-style ``<think>`` section even when
asked not to; that section is removed before validation and persistence.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
from pathlib import Path

import requests


SYSTEM = """You are a meticulous audiobook transcript editor. You receive two independent
ASR transcripts of the exact same audio interval. Return one faithful, verbatim transcript.
Resolve spelling, punctuation, and small word disagreements using both candidates and the
surrounding grammar. If one candidate contains spoken words that the other dropped, retain
them when they fit. Either ASR may hallucinate repeated phrases during music, sound effects,
or overlapping voices; remove those hallucinations instead of trying to merge them. Never
summarize, paraphrase, modernize, censor, invent dialogue, add speaker names, or describe
sound effects. Output only the corrected transcript, with no heading, notes, Markdown,
quotation fence, or reasoning."""

def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            record["id"]: record
            for line in handle
            if line.strip()
            for record in [json.loads(line)]
        }


def normalise(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def lexical_words(text: str) -> list[str]:
    return [normalise(token) for token in text.split() if normalise(token)]


def mechanical_merge(qwen: str, whisper: str) -> str:
    q_tokens = qwen.split()
    w_tokens = whisper.split()
    q_norm = [normalise(token) for token in q_tokens]
    w_norm = [normalise(token) for token in w_tokens]
    matcher = difflib.SequenceMatcher(a=q_norm, b=w_norm, autojunk=False)
    merged: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            merged.extend(w_tokens[j1:j2])
        elif tag == "delete":
            merged.extend(q_tokens[i1:i2])
        elif tag == "insert":
            merged.extend(w_tokens[j1:j2])
        else:
            q_part = q_tokens[i1:i2]
            w_part = w_tokens[j1:j2]
            merged.extend(q_part if len(q_part) > len(w_part) else w_part)
    return " ".join(merged)


def clean_generation(text: str) -> str:
    # Nemotron puts its internal reasoning in the normal content field. Keep
    # only the final channel after the last closing think tag.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return ""
    text = text.strip().strip("`").strip()
    text = re.sub(r"^(final transcript|corrected transcript|transcript)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([.!?][\"'’”)]?)(?=[A-Z])", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def valid_consensus(text: str, qwen: str, whisper: str) -> bool:
    words = lexical_words(text)
    q_words = lexical_words(qwen)
    w_words = lexical_words(whisper)
    shortest = min(len(q_words), len(w_words)) or 1
    longest = max(len(q_words), len(w_words)) or 1
    similarity = max(
        difflib.SequenceMatcher(a=words, b=q_words, autojunk=False).ratio(),
        difflib.SequenceMatcher(a=words, b=w_words, autojunk=False).ratio(),
    )
    return (
        bool(text)
        and 0.90 * shortest <= len(words) <= 1.05 * longest
        and similarity >= 0.70
        and "```" not in text
    )


def request_consensus(
    session: requests.Session,
    url: str,
    model: str,
    qwen: str,
    whisper: str,
    glossary: str,
    chapter_title: str,
    part: int,
) -> tuple[str, str]:
    glossary_block = (
        f"Reference spellings and domain terms:\n{glossary}\n\n" if glossary else ""
    )
    user = (
        "/no_think\nFINAL TRANSCRIPT ONLY. Do not explain your work.\n"
        f"Section: {chapter_title}, part {part}\n{glossary_block}"
        f"CANDIDATE A (Qwen ASR):\n{qwen}\n\n"
        f"CANDIDATE B (Whisper ASR):\n{whisper}\n\n"
        "Return only the faithful consensus transcript."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    for attempt in range(2):
        if attempt:
            payload["max_tokens"] = 8192
            payload["messages"][1]["content"] += "\nDo not include analysis; keep the answer within the candidate length."
        response = session.post(url, json=payload, timeout=(15, 900))
        response.raise_for_status()
        body = response.json()
        choice = body.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        consensus = clean_generation(content)
        if valid_consensus(consensus, qwen, whisper):
            return consensus, f"remote:{model}"
        if choice.get("finish_reason") != "length":
            break
    return mechanical_merge(qwen, whisper), "remote-mechanical-fallback"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("whisper", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default=os.environ.get("EDITOR_BASE_URL"))
    parser.add_argument("--model", default="nemotron-49b-v1_5")
    parser.add_argument(
        "--glossary-file",
        type=Path,
        help="optional UTF-8 file containing source-specific names or domain terms",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    qwen = load_jsonl(args.qwen)
    whisper = load_jsonl(args.whisper)
    existing = load_jsonl(args.output)
    pending = [chunk for chunk in manifest["chunks"] if chunk["id"] not in existing]
    print(f"Remote editor: {len(existing)} complete, {len(pending)} pending", flush=True)
    if not pending:
        return 0
    if not args.base_url:
        parser.error("--base-url is required (or set EDITOR_BASE_URL)")

    url = args.base_url.rstrip("/") + "/chat/completions"
    glossary = args.glossary_file.read_text(encoding="utf-8").strip() if args.glossary_file else ""
    session = requests.Session()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for chunk in pending:
            q_text = qwen[chunk["id"]]["text"].strip()
            w_text = whisper[chunk["id"]]["text"].strip()
            started = time.time()
            if lexical_words(q_text) == lexical_words(w_text):
                consensus, method = w_text, "identical"
            else:
                try:
                    consensus, method = request_consensus(
                        session,
                        url,
                        args.model,
                        q_text,
                        w_text,
                        glossary,
                        chunk["chapter_title"],
                        chunk["part"],
                    )
                except Exception as exc:  # keep the queue moving and auditable
                    print(f"remote error on {chunk['id']}: {exc}", flush=True)
                    consensus, method = mechanical_merge(q_text, w_text), "remote-error-fallback"
            record = {
                **chunk,
                "editor_model": args.model,
                "method": method,
                "similarity": round(
                    difflib.SequenceMatcher(
                        a=lexical_words(q_text), b=lexical_words(w_text), autojunk=False
                    ).ratio(),
                    6,
                ),
                "qwen_word_count": len(lexical_words(q_text)),
                "whisper_word_count": len(lexical_words(w_text)),
                "consensus_word_count": len(lexical_words(consensus)),
                "text": consensus,
                "elapsed_seconds": round(time.time() - started, 3),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            existing[chunk["id"]] = record
            print(
                f"[{len(existing):03d}/{len(manifest['chunks']):03d}] {chunk['id']} "
                f"{method} {record['consensus_word_count']} words in {record['elapsed_seconds']:.1f}s",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
