#!/usr/bin/env bash
set -euo pipefail

# Restart the parallel queue from checkpoints when configured jobs remain.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}
CONFIG=${1:-"$PROJECT_ROOT/config/subtitle_queue.json"}
[[ "$CONFIG" = /* ]] || CONFIG="$PROJECT_ROOT/$CONFIG"
PYTHON=${PYTHON:-python3}
INTERVAL_SECONDS=${INTERVAL_SECONDS:-300}
LOG=${LOG:-"$PROJECT_ROOT/work/queue_watchdog.log"}
QUEUE_SCRIPT="$SCRIPT_DIR/run_parallel_queued_books.sh"

log() {
  mkdir -p "$(dirname "$LOG")"
  printf '[queue-watchdog %s] %s\n' "$(date -u +%FT%TZ)" "$*" >> "$LOG"
}

active_pipeline_processes() {
  ps -eo stat=,cmd= | awk '
    $1 !~ /^Z/ &&
    ($0 ~ /run_(parallel_)?queued_books\.sh/ ||
     $0 ~ /transcribe_qwen\.py/ ||
     $0 ~ /transcribe_whisper\.py/ ||
     $0 ~ /reconcile_transcripts(_ollama|_vllm)?\.py/ ||
     $0 ~ /align_consensus\.py/ ||
     $0 ~ /build_subtitles\.py/ ||
     $0 ~ /mux_video\.py/) { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

all_complete() {
  "$PYTHON" "$SCRIPT_DIR/queue_config.py" all-complete "$CONFIG" "$PROJECT_ROOT" >/dev/null
}

log "started; interval=${INTERVAL_SECONDS}s"
while true; do
  if all_complete; then
    log 'all configured jobs complete; exiting'
    exit 0
  fi
  if ! active_pipeline_processes; then
    log 'no active queue worker found; restarting from checkpoints'
    setsid "$QUEUE_SCRIPT" "$CONFIG" >> "$PROJECT_ROOT/work/parallel_queue.log" 2>&1 < /dev/null &
    queue_pid=$!
    printf '%s\n' "$queue_pid" > "$PROJECT_ROOT/work/parallel_queue.pid"
    log "started queue pid $queue_pid"
  fi
  sleep "$INTERVAL_SECONDS"
done
