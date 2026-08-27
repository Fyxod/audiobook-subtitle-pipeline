import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_subtitles


class BuildSubtitlesTests(unittest.TestCase):
    def test_generic_basename_controls_generated_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            aligned = root / "aligned.jsonl"
            output = root / "output"
            manifest.write_text(
                json.dumps(
                    {
                        "chunks": [
                            {
                                "id": "c01_p01",
                                "chapter_id": 1,
                                "chapter_title": "Chapter 1",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            aligned.write_text(
                json.dumps(
                    {
                        "id": "c01_p01",
                        "chapter_id": 1,
                        "chapter_title": "Chapter 1",
                        "text": "A short public-domain sample.",
                        "surface_restored": True,
                        "words": [
                            {"text": "A", "start": 0.0, "end": 0.2},
                            {"text": "short", "start": 0.2, "end": 0.5},
                            {"text": "sample.", "start": 0.5, "end": 1.0},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "build_subtitles.py",
                str(manifest),
                str(aligned),
                str(output),
                "--basename",
                "sample_book",
            ]
            with patch.object(sys, "argv", argv):
                self.assertEqual(build_subtitles.main(), 0)
            self.assertTrue((output / "sample_book.en.srt").is_file())
            self.assertTrue((output / "sample_book.en.vtt").is_file())
            self.assertTrue((output / "sample_book.transcript.txt").is_file())
            self.assertTrue((output / "sample_book.word_timestamps.json").is_file())
            self.assertTrue((output / "subtitle_qc.json").is_file())


if __name__ == "__main__":
    unittest.main()
