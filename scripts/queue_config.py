#!/usr/bin/env python3
"""Validate batch configuration and expose shell-friendly queue records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def clean_field(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    value = value.strip()
    if "\t" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{label} must not contain tabs or newlines")
    return value


def resolve_path(root: Path, value: object, label: str) -> Path:
    path = Path(clean_field(value, label)).expanduser()
    return path if path.is_absolute() else root / path


def load_config(path: Path, root: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    jobs = data.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("config.jobs must be a non-empty array")

    normalized: list[dict] = []
    seen: set[str] = set()
    for index, raw in enumerate(jobs):
        if not isinstance(raw, dict):
            raise ValueError(f"jobs[{index}] must be an object")
        job_id = clean_field(raw.get("id"), f"jobs[{index}].id")
        if not SAFE_NAME.fullmatch(job_id):
            raise ValueError(f"jobs[{index}].id has unsafe characters")
        if job_id in seen:
            raise ValueError(f"duplicate job id: {job_id}")
        seen.add(job_id)
        basename = clean_field(
            raw.get("output_basename", job_id), f"jobs[{index}].output_basename"
        )
        if not SAFE_NAME.fullmatch(basename):
            raise ValueError(f"jobs[{index}].output_basename has unsafe characters")
        glossary_value = raw.get("glossary_file")
        glossary = (
            resolve_path(root, glossary_value, f"jobs[{index}].glossary_file")
            if glossary_value
            else None
        )
        normalized.append(
            {
                "id": job_id,
                "source": resolve_path(root, raw.get("source"), f"jobs[{index}].source"),
                "work_dir": resolve_path(
                    root, raw.get("work_dir", f"work/{job_id}"), f"jobs[{index}].work_dir"
                ),
                "output_dir": resolve_path(
                    root,
                    raw.get("output_dir", f"output/{job_id}"),
                    f"jobs[{index}].output_dir",
                ),
                "basename": basename,
                "glossary": glossary,
            }
        )
    return data, normalized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("jobs", "editor", "runtime", "all-complete"))
    parser.add_argument("config", type=Path)
    parser.add_argument("project_root", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    data, jobs = load_config(args.config.resolve(), root)
    if args.command == "jobs":
        for job in jobs:
            print(
                "\t".join(
                    (
                        job["id"],
                        str(job["source"]),
                        str(job["work_dir"]),
                        str(job["output_dir"]),
                        job["basename"],
                        str(job["glossary"]) if job["glossary"] else "",
                    )
                )
            )
        return 0

    if args.command == "editor":
        editor = data.get("editor", {})
        script = resolve_path(
            root,
            editor.get("script", "scripts/reconcile_transcripts.py"),
            "editor.script",
        )
        model = clean_field(editor.get("model", "Qwen/Qwen3-14B"), "editor.model")
        base_url_env = clean_field(
            editor.get("base_url_env", "EDITOR_BASE_URL"), "editor.base_url_env"
        )
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", base_url_env):
            raise ValueError("editor.base_url_env must be a valid environment variable name")
        print("\t".join((str(script), model, base_url_env)))
        return 0

    if args.command == "runtime":
        runtime = data.get("runtime", {})
        values = (
            int(runtime.get("qwen_batch_size", 2)),
            int(runtime.get("whisper_batch_size", 2)),
            int(runtime.get("whisper_input_group_size", 1)),
        )
        if any(value < 1 for value in values):
            raise ValueError("runtime batch sizes must be positive integers")
        print("\t".join(str(value) for value in values))
        return 0

    incomplete = [job["id"] for job in jobs if not (job["work_dir"] / "QUEUE_COMPLETE").is_file()]
    if incomplete:
        print(" ".join(incomplete))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
