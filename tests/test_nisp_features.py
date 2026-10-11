import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_nisp  # noqa: E402
import nisp_features as nf  # noqa: E402


FIELDS = list(nf.MANIFEST_COLUMNS)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class ManifestFixture:
    def __init__(self, root):
        self.root = root
        self.manifests = root / "manifests"
        self.manifests.mkdir()
        (root / "audio").mkdir()
        self.rows = [
            self.row("Hin_1", "train", 0),
            self.row("Hin_1", "train", 1),
            self.row("Kan_2", "val", 2),
            self.row("Tel_3", "test", 3),
        ]
        self.save()

    def row(self, speaker, split, index):
        return {
            "audio_path": str(self.root / "audio" / f"{speaker}_Eng_m_{index}.wav"),
            "speaker_id": speaker, "split": split, "height_cm": "170",
            "weight_kg": "70", "bmi": "24.221453", "age": "25",
            "sex": "Female" if speaker == "Hin_1" else "Male",
        }

    def save(self, status="ok"):
        write_csv(self.manifests / "all.csv", FIELDS, self.rows)
        for split in ("train", "val", "test"):
            write_csv(self.manifests / f"{split}.csv", FIELDS,
                      [row for row in self.rows if row["split"] == split])
        speakers = []
        for speaker in sorted({row["speaker_id"] for row in self.rows}):
            records = [row for row in self.rows if row["speaker_id"] == speaker]
            first = records[0]
            speakers.append({
                "speaker_id": speaker,
                "official_split": "test" if first["split"] == "test" else "train",
                **{field: first[field] for field in ("split", "height_cm", "weight_kg", "bmi", "age", "sex")},
                "num_recordings": len(records),
            })
        write_csv(self.manifests / "speakers.csv",
                  ["speaker_id", "official_split", "split", "height_cm", "weight_kg",
                   "bmi", "age", "sex", "num_recordings"], speakers)
        audit = {
            "status": status, "manifests_written": status == "ok",
            "counts_by_split": {
                split: {"recordings": sum(row["split"] == split for row in self.rows)}
                for split in ("train", "val", "test")
            },
        }
        (self.manifests / "audit.json").write_text(json.dumps(audit), encoding="utf-8")


class FeatureContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.fixture = ManifestFixture(self.root)

    def test_manifest_validation_is_lexical_and_disjoint(self):
        rows, speakers, fingerprint = nf.load_manifests(self.root)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(speakers), 3)
        self.assertEqual(len(fingerprint), 64)
        self.assertTrue(all("source_key" in row for row in rows))

    def test_stale_audit_and_mixed_labels_are_rejected(self):
        self.fixture.save(status="preparing")
        with self.assertRaisesRegex(nf.FeatureError, "stale"):
            nf.load_manifests(self.root)
        self.fixture.rows[1]["bmi"] = "99"
        self.fixture.save()
        with self.assertRaisesRegex(nf.FeatureError, "mixed labels"):
            nf.load_manifests(self.root)

    def test_balanced_pilot_is_train_only_and_deterministic(self):
        rows = []
        for prefix in nf.PREFIXES:
            for index in range(25):
                rows.append({
                    "audio_path": f"/audio/{prefix}_{index:04d}.wav",
                    "speaker_id": f"{prefix}_{index:04d}", "split": "train",
                    "source_index": str(len(rows)), "source_key": f"{len(rows):064x}",
                })
        rows.append({"audio_path": "/audio/test.wav", "speaker_id": "Hin_9999",
                     "split": "test", "source_index": "999", "source_key": "f" * 64})
        first = nf.select_rows(rows, 100)
        second = nf.select_rows(rows, 100)
        self.assertEqual([row["source_key"] for row in first],
                         [row["source_key"] for row in second])
        self.assertTrue(all(row["split"] == "train" for row in first))
        self.assertEqual(
            {prefix: sum(row["speaker_id"].startswith(prefix + "_") for row in first)
             for prefix in nf.PREFIXES},
            {prefix: 20 for prefix in nf.PREFIXES},
        )

    def test_all_layers_are_shape_checked_finite_and_aligned(self):
        value = nf.validate_layer_stack(np.zeros((12, 768), dtype=np.float32))
        self.assertEqual(value.shape, (12, 768))
        bad = np.zeros((12, 768), dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(nf.FeatureError, "NaN"):
            nf.validate_layer_stack(bad)
        states = [np.zeros((1, 4, 768), dtype=np.float32) for _ in range(13)]
        extract_nisp.validate_hidden_state_contract(states, states[-1], np.allclose)
        states[-1] = np.ones((1, 4, 768), dtype=np.float32)
        with self.assertRaisesRegex(nf.FeatureError, "does not match"):
            extract_nisp.validate_hidden_state_contract(
                states, np.zeros((1, 4, 768), dtype=np.float32), np.allclose
            )

    def test_equal_speaker_average_not_weighted_by_other_speakers(self):
        rows = [
            {"speaker_id": "Hin_1", "split": "train", "height_cm": "1", "weight_kg": "1",
             "bmi": "1", "age": "1", "sex": "Female"},
            {"speaker_id": "Hin_1", "split": "train", "height_cm": "1", "weight_kg": "1",
             "bmi": "1", "age": "1", "sex": "Female"},
            {"speaker_id": "Kan_2", "split": "val", "height_cm": "2", "weight_kg": "2",
             "bmi": "2", "age": "2", "sex": "Male"},
        ]
        ssl = np.zeros((3, 12, 768), dtype=np.float32)
        ssl[0] = 0
        ssl[1] = 2
        ssl[2] = 9
        mfcc = np.asarray([[0] * 26, [2] * 26, [9] * 26], dtype=np.float32)
        meta, by_speaker, by_mfcc = nf.aggregate_speakers(rows, ssl, mfcc)
        self.assertEqual([row["num_recordings"] for row in meta], ["2", "1"])
        self.assertTrue(np.allclose(by_speaker[0], 1))
        self.assertTrue(np.allclose(by_mfcc[0], 1))
        self.assertTrue(np.allclose(by_speaker[1], 9))

    def test_config_change_and_corrupt_checkpoint_are_rejected(self):
        rows, _speakers, fingerprint = nf.load_manifests(self.root)
        config = nf.build_config(fingerprint, rows, None, 10.0, 100)
        output = self.root / "output"
        output.mkdir()
        extract_nisp._verify_or_write_config(output, config)
        changed = dict(config)
        changed["revision"] = "changed"
        with self.assertRaisesRegex(nf.FeatureError, "configuration"):
            extract_nisp._verify_or_write_config(output, changed)

        chunk = output / "chunk.npz"
        nf.atomic_npz(
            chunk,
            source_index=np.asarray([int(row["source_index"]) for row in rows]),
            source_key=np.asarray([row["source_key"] for row in rows], dtype="U64"),
            ssl=np.zeros((len(rows), 12, 768), dtype=np.float32),
            mfcc=np.zeros((len(rows), 26), dtype=np.float32),
            source_rate=np.full(len(rows), 48000), source_frames=np.ones(len(rows), dtype=np.int64),
            target_frames=np.ones(len(rows), dtype=np.int64),
            feature_frames=np.ones(len(rows), dtype=np.int64),
        )
        nf.validate_chunk(chunk, rows)
        chunk.write_bytes(chunk.read_bytes()[:20])
        with self.assertRaisesRegex(nf.FeatureError, "checkpoint"):
            nf.validate_chunk(chunk, rows)

    def test_fully_cached_resume_reports_reuse_without_forward_pass(self):
        rows, speakers, fingerprint = nf.load_manifests(self.root)
        config = nf.build_config(fingerprint, rows, None, 10.0, 100)
        output = self.root / "cached-output"
        chunk = output / "chunks/chunk-000000.npz"
        count = len(rows)
        nf.atomic_npz(
            chunk,
            source_index=np.asarray([int(row["source_index"]) for row in rows], dtype=np.int64),
            source_key=np.asarray([row["source_key"] for row in rows], dtype="U64"),
            ssl=np.zeros((count, 12, 768), dtype=np.float32),
            mfcc=np.zeros((count, 26), dtype=np.float32),
            source_rate=np.full(count, 48000, dtype=np.int32),
            source_frames=np.full(count, 48000, dtype=np.int64),
            target_frames=np.full(count, 16000, dtype=np.int64),
            feature_frames=np.full(count, 49, dtype=np.int64),
        )
        fake_torch = SimpleNamespace(
            __version__="test",
            version=SimpleNamespace(cuda="test"),
            cuda=SimpleNamespace(max_memory_allocated=lambda: 0),
        )
        runtime = {"gpu_name": "NVIDIA H200", "versions": {}, "torch": fake_torch}
        args = SimpleNamespace(
            output=output, model_dir=self.root, max_seconds=10.0, chunk_size=100
        )
        with patch.dict("os.environ", {"SLURM_JOB_ID": "cached-test"}, clear=False):
            with patch.object(extract_nisp, "_load_runtime", return_value=runtime):
                with patch.object(extract_nisp, "_extract_one") as forward:
                    metadata = extract_nisp.run_extraction(
                        args, {"selected": rows, "speaker_rows": speakers, "config": config}
                    )
        forward.assert_not_called()
        self.assertEqual(metadata["attempted"], 0)
        self.assertEqual(metadata["computed_this_attempt"], 0)
        self.assertEqual(metadata["reused_records"], count)
        self.assertEqual(metadata["completed"], count)
        self.assertEqual(metadata["success"], count)


if __name__ == "__main__":
    unittest.main()
