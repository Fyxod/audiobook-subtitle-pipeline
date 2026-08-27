# Report schemas

These schemas document the small technical reports produced by the pipeline without publishing source-derived values:

- `subtitle_qc.schema.json` describes cue, timing, and alignment checks emitted by `build_subtitles.py`.
- `verification.schema.json` describes the FFprobe summary emitted by `mux_video.py`.

Actual run reports belong under `reports/runs/`, which is ignored because filenames, durations, counts, metadata, and hashes can identify private source media.
