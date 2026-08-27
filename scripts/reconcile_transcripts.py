#!/usr/bin/env python3
"""Reconcile Qwen and Whisper transcripts with a local instruction model."""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM = """You are a meticulous audiobook transcript editor. You receive two independent
ASR transcripts of the exact same audio interval. Return one faithful, verbatim transcript.
Resolve spelling, punctuation, and small word disagreements using both candidates and the
surrounding grammar. If one candidate contains spoken words that the other dropped, retain
them when they fit. Either ASR may hallucinate repeated phrases during music, sound effects,
or overlapping voices; remove those hallucinations instead of trying to merge them. Never
summarize, paraphrase, modernize, censor, invent dialogue, add
speaker names, or describe sound effects. Output only the corrected transcript, with no
heading, notes, Markdown, or quotation fence."""

def load_jsonl(path: Path) -> dict[str, dict]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record
    return records


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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip().strip("`").strip()
    text = re.sub(r"^(corrected transcript|transcript)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"([.!?][\"'’”)]?)(?=[A-Z])", r"\1 ", text)
    return re.sub(r"\s+", " ", text).strip()


def valid_consensus(text: str, qwen: str, whisper: str) -> bool:
    words = lexical_words(text)
    count = len(words)
    q_words = lexical_words(qwen)
    w_words = lexical_words(whisper)
    q_count = len(q_words)
    w_count = len(w_words)
    shortest = min(q_count, w_count) or 1
    longest = max(q_count, w_count) or 1
    similarity = max(
        difflib.SequenceMatcher(a=words, b=q_words, autojunk=False).ratio(),
        difflib.SequenceMatcher(a=words, b=w_words, autojunk=False).ratio(),
    )
    return (
        bool(text)
        and 0.90 * shortest <= count <= 1.05 * longest
        and similarity >= 0.70
        and "```" not in text
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("qwen", type=Path)
    parser.add_argument("whisper", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument(
        "--glossary-file",
        type=Path,
        help="optional UTF-8 file containing source-specific names or domain terms",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    qwen = load_jsonl(args.qwen)
    whisper = load_jsonl(args.whisper)
    existing = load_jsonl(args.output) if args.output.exists() else {}
    pending = [c for c in manifest["chunks"] if c["id"] not in existing]
    print(f"Consensus: {len(existing)} complete, {len(pending)} pending", flush=True)
    if not pending:
        return 0

    model_id = args.model
    glossary = args.glossary_file.read_text(encoding="utf-8").strip() if args.glossary_file else ""
    glossary_block = (
        f"Reference spellings and domain terms:\n{glossary}\n\n" if glossary else ""
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output:
        for chunk in pending:
            q_text = qwen[chunk["id"]]["text"].strip()
            w_text = whisper[chunk["id"]]["text"].strip()
            q_words = lexical_words(q_text)
            w_words = lexical_words(w_text)
            similarity = difflib.SequenceMatcher(a=q_words, b=w_words, autojunk=False).ratio()
            started = time.time()

            if q_words == w_words:
                consensus = w_text
                method = "identical"
            else:
                user = (
                    f"Section: {chunk['chapter_title']}, part {chunk['part']}\n"
                    f"{glossary_block}CANDIDATE A (Qwen ASR):\n{q_text}\n\n"
                    f"CANDIDATE B (Whisper ASR):\n{w_text}\n\n"
                    "Return only the faithful consensus transcript."
                )
                messages = [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=1800,
                        do_sample=False,
                        repetition_penalty=1.01,
                    )
                generated = tokenizer.decode(
                    output_ids[0, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                )
                consensus = clean_generation(generated)
                method = "qwen3-14b"
                if not valid_consensus(consensus, q_text, w_text):
                    consensus = mechanical_merge(q_text, w_text)
                    method = "mechanical-fallback"

            record = {
                **chunk,
                "editor_model": model_id,
                "method": method,
                "similarity": round(similarity, 6),
                "qwen_word_count": len(q_words),
                "whisper_word_count": len(w_words),
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
                f"sim={similarity:.3f} {method} {record['consensus_word_count']} words",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
