#!/usr/bin/env bash
set -euo pipefail

# Run configured inputs sequentially on one local ASR/alignment device.

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

log() { printf '[queue %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }

run_editor() {
  local work="$1" glossary="$2"
  local editor_name editor_url
  editor_name=$(basename "$EDITOR_SCRIPT")
  local -a command=(
    "$PYTHON" "$EDITOR_SCRIPT"
    "$work/chunk_manifest.json"
    "$work/qwen_transcription.jsonl"
    "$work/whisper_transcription.jsonl"
    "$work/consensus.jsonl"
    --model "$EDITOR_MODEL"
  )
  if [[ "$editor_name" != "reconcile_transcripts.py" ]]; then
    editor_url=${!EDITOR_URL_ENV:-}
    [[ -n "$editor_url" ]] || {
      log "set $EDITOR_URL_ENV before using $editor_name"
      return 1
    }
    command+=(--base-url "$editor_url")
  fi
  [[ -n "$glossary" ]] && command+=(--glossary-file "$glossary")
  "${command[@]}"
}

run_job() {
  local job_id="$1" source="$2" work="$3" output="$4" output_basename="$5" glossary="$6"
  local probe="$work/ffprobe.json"
  local wav="$work/audio_16k_mono.wav"
  local channels
  local -a qwen_command

  if [[ -e "$work/QUEUE_COMPLETE" ]]; then
    log "$job_id already complete; skipping"
    return
  fi
  [[ -f "$source" ]] || {
    log "$job_id source does not exist: $source"
    return 1
  }
  [[ -z "$glossary" || -f "$glossary" ]] || {
    log "$job_id glossary does not exist: $glossary"
    return 1
  }

  mkdir -p "$work" "$output"
  log "$job_id: preparing source"
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
  run_editor "$work" "$glossary"
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

for row in "${JOB_ROWS[@]}"; do
  IFS=$'\t' read -r job_id source work output output_basename glossary <<< "$row"
  run_job "$job_id" "$source" "$work" "$output" "$output_basename" "$glossary"
done
log 'all configured jobs complete'
