# Audiobook Subtitle Pipeline

A checkpointed pipeline for converting chaptered, long-form audio into reconciled transcripts, word-aligned SRT/VTT subtitles, QC reports, and an optional MKV with copied source audio and a selectable subtitle track.

## Pipeline

1. `prepare_chunks.py` converts FFprobe chapter metadata and a mono 16 kHz working WAV into low-energy, chapter-aware chunks.
2. `transcribe_qwen.py` runs Qwen3-ASR with Qwen3-ForcedAligner and saves append-only word-timestamp checkpoints.
3. `transcribe_whisper.py` runs Whisper Large-v3 as an independent transcription and timing pass.
4. A reconciliation script adjudicates wording differences with a local model, an OpenAI-compatible endpoint, or Ollama.
5. `align_consensus.py` force-aligns the accepted transcript; `fuse_word_timestamps.py` repairs timings only where independent anchors agree locally.
6. `build_subtitles.py` writes SRT, VTT, a plain transcript, word timestamps, and subtitle QC metrics.
7. `mux_video.py` optionally creates a black H.264 video with copied source audio, preserved chapters, and selectable subtitles.

## Why two transcription passes?

Qwen supplies the primary transcript and word alignment. Whisper provides an independent wording and timing signal. Reconciliation chooses a faithful transcript from both candidates, while timestamp fusion accepts Whisper anchors only when nearby matches show a consistent offset. This limits damage from insertions, omissions, or overlap errors in either model.

## Checkpointing / reliability

Every JSONL stage is keyed by chunk ID and skips completed records on restart. Queue runners preserve intermediate files, retry incomplete reconciliation records, and write `QUEUE_COMPLETE` only after subtitles, muxing, checksums, and verification finish.

## Usage

Requirements: Python 3.10+, FFmpeg/FFprobe, a CUDA-capable PyTorch installation, and enough GPU memory for the selected models.

```bash
git clone https://github.com/Fyxod/audiobook-subtitle-pipeline.git
cd audiobook-subtitle-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PROJECT_ROOT="$(pwd)"
export WORK_DIR="$PROJECT_ROOT/work/sample_book"
mkdir -p "$PROJECT_ROOT/source" "$WORK_DIR" "$PROJECT_ROOT/output/sample_book"

# Put an audio file you have rights to at source/sample_book.m4b.
ffprobe -v error -show_streams -show_chapters -of json \
  "$PROJECT_ROOT/source/sample_book.m4b" > "$WORK_DIR/ffprobe.json"
ffmpeg -v error -i "$PROJECT_ROOT/source/sample_book.m4b" -map 0:a:0 \
  -ar 16000 -ac 1 -c:a pcm_s16le "$WORK_DIR/audio_16k_mono.wav"

python scripts/prepare_chunks.py "$WORK_DIR/audio_16k_mono.wav" \
  "$WORK_DIR/ffprobe.json" "$WORK_DIR/chunk_manifest.json"
python scripts/transcribe_qwen.py "$WORK_DIR/chunk_manifest.json" \
  "$WORK_DIR/qwen_transcription.jsonl"
python scripts/transcribe_whisper.py "$WORK_DIR/chunk_manifest.json" \
  "$WORK_DIR/whisper_transcription.jsonl"
python scripts/reconcile_transcripts.py "$WORK_DIR/chunk_manifest.json" \
  "$WORK_DIR/qwen_transcription.jsonl" "$WORK_DIR/whisper_transcription.jsonl" \
  "$WORK_DIR/consensus.jsonl"
python scripts/align_consensus.py "$WORK_DIR/chunk_manifest.json" \
  "$WORK_DIR/consensus.jsonl" "$WORK_DIR/final_aligned.jsonl"
python scripts/fuse_word_timestamps.py "$WORK_DIR/final_aligned.jsonl" \
  "$WORK_DIR/qwen_transcription.jsonl" "$WORK_DIR/whisper_transcription.jsonl" \
  "$WORK_DIR/final_aligned_fused.jsonl"
python scripts/build_subtitles.py "$WORK_DIR/chunk_manifest.json" \
  "$WORK_DIR/final_aligned_fused.jsonl" "$PROJECT_ROOT/output/sample_book" \
  --basename sample_book
```

For batch jobs, edit `config/subtitle_queue.json`, set `EDITOR_BASE_URL` when using a remote editor, and run `scripts/run_queued_books.sh` or `scripts/run_parallel_queued_books.sh`. See `docs/PIPELINE.md` for endpoint and watchdog details. A project-specific glossary can be supplied with `--glossary-file`; none is embedded in the code.

## Repository structure

```text
config/    data-driven batch job example
docs/      operational runbook and recovery notes
reports/   QC and media-verification report schemas
scripts/   preparation, ASR, reconciliation, alignment, subtitle, and mux tools
```

## Notes

Source audio, generated transcripts/subtitles, model caches, runtime logs, checkpoints, hashes, and final media are deliberately excluded from Git. The sample config documents paths but does not include media.

Only process audio for which you have the necessary rights. Source media and generated copyrighted content are not included in this repository.
