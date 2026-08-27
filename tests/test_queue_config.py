import json
import tempfile
import unittest
from pathlib import Path

from scripts.queue_config import load_config


class QueueConfigTests(unittest.TestCase):
    def write_config(self, root: Path, jobs: list[dict]) -> Path:
        path = root / "queue.json"
        path.write_text(json.dumps({"jobs": jobs}), encoding="utf-8")
        return path

    def test_paths_and_defaults_are_data_driven(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_config(
                root,
                [{"id": "volume_a", "source": "source/input.m4b"}],
            )
            _, jobs = load_config(path, root)
            self.assertEqual(jobs[0]["work_dir"], root / "work" / "volume_a")
            self.assertEqual(jobs[0]["output_dir"], root / "output" / "volume_a")
            self.assertEqual(jobs[0]["basename"], "volume_a")

    def test_duplicate_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_config(
                root,
                [
                    {"id": "duplicate", "source": "source/a.m4b"},
                    {"id": "duplicate", "source": "source/b.m4b"},
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate job id"):
                load_config(path, root)

    def test_unsafe_output_basename_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self.write_config(
                root,
                [
                    {
                        "id": "volume_a",
                        "source": "source/input.m4b",
                        "output_basename": "../outside",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "unsafe characters"):
                load_config(path, root)


if __name__ == "__main__":
    unittest.main()
