import csv
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location("gpu_smoke", Path(__file__).parents[1] / "gpu_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "manifests").mkdir()
        (self.root / "audio").mkdir()
        (self.root / "manifests/audit.json").write_text(
            json.dumps({"status": "ok", "manifests_written": True}))
        self.rows = []
        for prefix in smoke.PREFIXES:
            for index in (1, 2):
                path = self.root / "audio" / (prefix + "_0001_Eng_m_" + str(index) + ".wav")
                path.touch()
                self.rows.append({"audio_path": str(path), "speaker_id": prefix + "_0001", "split": "train"})

    def save(self):
        with (self.root / "manifests/train.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=("audio_path", "speaker_id", "split"))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_exactly_one_training_clip_per_language(self):
        self.save()
        clips = smoke.select_clips(self.root)
        self.assertEqual([row["speaker_id"][:3] for row in clips], list(smoke.PREFIXES))
        self.assertTrue(all(row["audio_path"].endswith("_1.wav") for row in clips))

    def test_test_split_contamination_rejected(self):
        self.rows[-1]["split"] = "test"
        self.save()
        with self.assertRaisesRegex(ValueError, "Non-training"):
            smoke.select_clips(self.root)

    def test_missing_language_rejected(self):
        self.rows = self.rows[:-2]
        self.save()
        with self.assertRaisesRegex(ValueError, "Missing training languages"):
            smoke.select_clips(self.root)

    def test_outside_audio_path_rejected(self):
        self.rows[0]["audio_path"] = str(self.root / "outside_Eng_.wav")
        self.save()
        with self.assertRaises(ValueError):
            smoke.select_clips(self.root)

    def test_stale_audit_rejected(self):
        self.save()
        (self.root / "manifests/audit.json").write_text('{"status":"preparing"}')
        with self.assertRaisesRegex(ValueError, "audit"):
            smoke.select_clips(self.root)

    def test_login_execution_rejected_before_ml_imports(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SLURM_JOB_ID"):
                smoke.run(SimpleNamespace(), [])


if __name__ == "__main__":
    unittest.main()
