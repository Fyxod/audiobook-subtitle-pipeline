# Operational runbook

## Checkpoint model

Each configured job has its own working directory. The chunk manifest is the source of truth for chunk IDs; transcription, reconciliation, alignment, and fusion are append-only JSONL checkpoints keyed by those IDs. Restarting a stage skips records already present.

Do not delete an entire checkpoint to recover from a transient failure. The parallel runner backs up mechanically merged reconciliation records and retries only those records when the editor becomes available.

## Batch configuration

`config/subtitle_queue.json` contains a `jobs` array. Every job defines a stable ID, exact source file, work directory, output directory, output basename, and optional glossary file. Paths may be absolute or relative to the repository root. Add, remove, or reorder jobs in the array; the runners do not assume a fixed number of inputs.

The `editor` block selects a reconciliation script and model. Endpoint addresses are not stored in the repository. Export the variable named by `base_url_env` before starting a remote editor job:

```bash
export EDITOR_BASE_URL="http://your-editor-host:port/v1"
scripts/run_queued_books.sh config/subtitle_queue.json
```

For Ollama, set `editor.script` to `scripts/reconcile_transcripts_ollama.py`, choose a model available on that server, and set the endpoint to the server root. For local reconciliation, set the script to `scripts/reconcile_transcripts.py`; no endpoint variable is required.

## Parallel queue operation

The parallel runner overlaps remote transcript reconciliation for one input with local ASR for the next input. It expects the configured editor to be reachable and uses the same checkpoints as the sequential runner.

```bash
export PROJECT_ROOT="$(pwd)"
export EDITOR_BASE_URL="http://your-editor-host:port/v1"
setsid scripts/run_parallel_queued_books.sh config/subtitle_queue.json \
  >> "$PROJECT_ROOT/work/parallel_queue.log" 2>&1 < /dev/null &
echo $! > "$PROJECT_ROOT/work/parallel_queue.pid"

setsid scripts/monitor_subtitle_queue.sh config/subtitle_queue.json \
  > "$PROJECT_ROOT/work/queue_watchdog.stderr" 2>&1 < /dev/null &
echo $! > "$PROJECT_ROOT/work/queue_watchdog.pid"
```

Set `PYTHON` to override the Python executable. Set `HF_HOME`, `HF_HUB_OFFLINE`, and `TRANSFORMERS_OFFLINE` when models are stored in a custom location or the processing host must remain offline.

## Inspecting progress

```bash
ps -eo pid,ppid,stat,etime,cmd | rg \
  'run_.*queued|transcribe_|reconcile_transcripts|align_consensus|build_subtitles|mux_video'

find work -name QUEUE_COMPLETE -print
find work -name '*.jsonl' -exec wc -l {} +
```

`QUEUE_COMPLETE` is written only after alignment, timing fusion, subtitle generation, muxing, checksums, and the final FFprobe verification file.

## Pause and resume

To pause, terminate the queue, watchdog, and their child workers. Keep all job working directories. Running the same command later resumes from the checkpoint files. The watchdog checks for unfinished configured jobs and restarts the parallel runner only when no pipeline worker is active.

## Operational notes

- A source must contain chapter metadata; chunk boundaries remain inside chapters and are moved toward locally quiet audio.
- Multi-channel inputs are converted to a mono 16 kHz working file. Original audio is copied, not re-encoded, when the MKV is created.
- Remote editors receive transcript text for reconciliation. Use the local editor when source text must not leave the processing host.
- Keep output media, runtime logs, endpoints, glossaries, and checkpoint data outside Git. Generated QC and verification JSON belong under `reports/runs/`, which is ignored.
