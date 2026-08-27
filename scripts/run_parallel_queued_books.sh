#!/usr/bin/env bash
set -euo pipefail

# Pipeline configured jobs by overlapping remote reconciliation with local ASR.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
CONFIG=${1:-"$PROJECT_ROOT/config/subtitle_queue.json"}
[[ "$CONFIG" = /* ]] || CONFIG="$PROJECT_ROOT/$CONFIG"
PYTHON=${PYTHON:-python3}

cd "$PROJECT_ROOT"
export HF_HOME=${HF_HOME:-"$PROJECT_ROOT/models/huggingface"}
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}

mapfile -t JOB_ROWS < <("$PYTHON" scripts/queue_config.py jobs "$CONFIG" "$PROJECT_ROOT")
IFS=$'\t' read -r EDITOR_SCRIPT EDITOR_MODEL EDITOR_URL_ENV < <(
  "$PYTHON" scripts/queue_config.py editor "$CONFIG" "$PROJECT_ROOT"
)
IFS=$'\t' read -r QWEN_BATCH WHISPER_BATCH WHISPER_GROUP < <(
  "$PYTHON" scripts/queue_config.py runtime "$CONFIG" "$PROJECT_ROOT"
)
EDITOR_NAME=$(basename "$EDITOR_SCRIPT")
[[ "$EDITOR_NAME" != "reconcile_transcripts.py" ]] || {
  printf 'parallel runner requires a remote editor; use run_queued_books.sh for local reconciliation\n' >&2
  exit 2
}
EDITOR_URL=${!EDITOR_URL_ENV:-}
[[ -n "$EDITOR_URL" ]] || {
  printf 'set %s before starting the parallel runner\n' "$EDITOR_URL_ENV" >&2
  exit 2
}

log() { printf '[parallel-queue %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

declare -A EDITOR_PIDS=()

prepare_local_asr() {
  local job_id="$1" source="$2" work="$3" output="$4" glossary="$5"
  local probe="$work/ffprobe.json"
  local wav="$work/audio_16k_mono.wav"
  local channels
  local -a qwen_command

  [[ -f "$source" ]] || {
    log "$job_id source does not exist: $source"
    return 1
  }
  [[ -z "$glossary" || -f "$glossary" ]] || {
    log "$job_id glossary does not exist: $glossary"
    return 1
  }
  mkdir -p "$work" "$output"
  log "$job_id: preparing local ASR"
  if [[ ! -s "$probe" ]]; then
    ffprobe -v error -show_streams -show_chapters -of json "$source" > "$probe"
  fi
  if [[ ! -s "$wav" ]]; then
    channels=$(ffprobe -v error -select_streams a:0 -show_entries stream=channels -of csv=p=0 "$source")
    if [[ "$channels" -ge 3 ]]; then
      ffmpeg -y -v error -i "$source" -map 0:a:0 -af 'pan=mono|c0=FC' -ar 16000 -ac 1 -c:a pcm_s16le "$wav"
    else
      ffmpeg -y -v error -i "$source" -map 0:a:0 -ar 16000 -ac 1 -c:a pcm_s16le "$wav"
    fi
  fi
  if [[ ! -s "$work/chunk_manifest.json" ]]; then
    "$PYTHON" scripts/prepare_chunks.py "$wav" "$probe" "$work/chunk_manifest.json"
  fi
  qwen_command=(
    "$PYTHON" scripts/transcribe_qwen.py
    "$work/chunk_manifest.json" "$work/qwen_transcription.jsonl"
    --batch-size "$QWEN_BATCH"
  )
  [[ -n "$glossary" ]] && qwen_command+=(--context-file "$glossary")
  "${qwen_command[@]}"
  "$PYTHON" scripts/transcribe_whisper.py +    "$work/chunk_manifest.json" "$work/whisper_transcription.jsonl" +    --batch-size "$WHISPER_BATCH" --input-group-size "$WHISPER_GROUP"
}

remove_retryable_fallbacks() {
  local checkpoint="$1"
  [[ -s "$checkpoint" ]] || return 0
  "$PYTHON" - "$checkpoint" <<'PY'
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
retry = [record for record in records if "fallback" in str(record.get("method", "")).lower()]
if retry:
    backup = path.with_name(path.stem + ".fallback_backup" + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
    keep = [record for record in records if record not in retry]
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        for record in keep:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary, path)
    print(f"removed {len(retry)} fallback records for retry", flush=True)
PY
}

start_remote_reconcile() {
  local job_id="$1" work="$2" glossary="$3"
  local -a command=(
    "$PYTHON" "$EDITOR_SCRIPT"
    "$work/chunk_manifest.json"
    "$work/qwen_transcription.jsonl"
    "$work/whisper_transcription.jsonl"
    "$work/consensus.jsonl"
    --base-url "$EDITOR_URL"
    --model "$EDITOR_MODEL"
  )
  [[ -n "$glossary" ]] && command+=(--glossary-file "$glossary")
  remove_retryable_fallbacks "$work/consensus.jsonl"
  log "$job_id: starting remote reconciliation ($EDITOR_MODEL)"
  setsid "${command[@]}" > "$work/remote_reconcile.log" 2>&1 < /dev/null &
  EDITOR_PIDS["$job_id"]=$!
  printf '%s\n' "${EDITOR_PIDS[$job_id]}" > "$work/remote_reconcile.pid"
}

finish_remote_reconcile() {
  local job_id="$1" work="$2"
  local editor_pid="${EDITOR_PIDS[$job_id]:-}"
  local count expected
  if [[ -n "$editor_pid" ]]; then
    log "$job_id: waiting for remote reconciliation"
    wait "$editor_pid"
    unset 'EDITOR_PIDS[$job_id]'
  fi
  count=$(wc -l < "$work/consensus.jsonl")
  expected=$("$PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))["chunks"]))' "$work/chunk_manifest.json")
  [[ "$count" -eq "$expected" ]] || {
    log "$job_id consensus incomplete: $count/$expected"
    return 1
  }
}

finalize_job() {
  local job_id="$1" source="$2" work="$3" output="$4" output_basename="$5"
  "$PYTHON" scripts/align_consensus.py +    "$work/chunk_manifest.json" "$work/consensus.jsonl" "$work/final_aligned.jsonl" +    --batch-size "$QWEN_BATCH"
  "$PYTHON" scripts/fuse_word_timestamps.py +    "$work/final_aligned.jsonl" "$work/qwen_transcription.jsonl" +    "$work/whisper_transcription.jsonl" "$work/final_aligned_fused.jsonl"
  "$PYTHON" scripts/build_subtitles.py +    "$work/chunk_manifest.json" "$work/final_aligned_fused.jsonl" "$output" +    --basename "$output_basename"
  "$PYTHON" scripts/mux_video.py +    "$source" "$output/$output_basename.en.srt" +    "$output/${output_basename}_with_subtitles.mkv"
  (
    cd "$output"
    sha256sum +      "$output_basename.en.srt" +      "$output_basename.en.vtt" +      "${output_basename}_with_subtitles.mkv" > SHA256SUMS
  )
  touch "$work/QUEUE_COMPLETE"
  log "$job_id complete"
}

pending_rows=()
for row in "${JOB_ROWS[@]}"; do
  IFS=$'\t' read -r job_id source work output output_basename glossary <<< "$row"
  if [[ -e "$work/QUEUE_COMPLETE" ]]; then
    log "$job_id already complete; skipping"
  else
    pending_rows+=("$row")
  fi
done
if [[ "${#pending_rows[@]}" -eq 0 ]]; then
  log 'all configured jobs already complete'
  exit 0
fi

previous_row=""
for row in "${pending_rows[@]}"; do
  IFS=$'\t' read -r job_id source work output output_basename glossary <<< "$row"
  prepare_local_asr "$job_id" "$source" "$work" "$output" "$glossary"
  start_remote_reconcile "$job_id" "$work" "$glossary"
  if [[ -n "$previous_row" ]]; then
    IFS=$'\t' read -r previous_id previous_source previous_work previous_output previous_basename previous_glossary <<< "$previous_row"
    finish_remote_reconcile "$previous_id" "$previous_work"
    finalize_job "$previous_id" "$previous_source" "$previous_work" "$previous_output" "$previous_basename"
  fi
  previous_row="$row"
done

IFS=$'\t' read -r previous_id previous_source previous_work previous_output previous_basename previous_glossary <<< "$previous_row"
finish_remote_reconcile "$previous_id" "$previous_work"
finalize_job "$previous_id" "$previous_source" "$previous_work" "$previous_output" "$previous_basename"
log 'all configured jobs complete'
