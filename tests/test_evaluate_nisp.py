import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_nisp as evaluation  # noqa: E402
import nisp_features as nf  # noqa: E402


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def synthetic_rows(test_bmi="poisoned-test-label"):
    definitions = [
        ("Hin_1", "train", "20", "Female", "20"),
        ("Kan_2", "train", "40", "Male", "30"),
        ("Mal_3", "val", "25", "Female", "22"),
        ("Tam_4", "val", "35", "Male", "28"),
        ("Tel_5", "test", "30", "Female", test_bmi),
    ]
    return [
        {"speaker_id": speaker, "split": split, "height_cm": "170", "weight_kg": "70",
         "bmi": bmi, "age": age, "sex": sex, "language": speaker[:3],
         "num_recordings": "1"}
        for speaker, split, age, sex, bmi in definitions
    ]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ssl = np.zeros((5, 12, 768), dtype=np.float32)
        self.mfcc = np.zeros((5, 26), dtype=np.float32)
        for index in range(5):
            self.ssl[index, 11, 0] = index
            self.mfcc[index, 0] = index

    def make_bundle(self, limit=None, completed=5):
        feature_dir = self.root / ("features-" + str(len(list(self.root.iterdir()))))
        feature_dir.mkdir()
        config = {
            "limit": limit,
            "selection_inventory": [{"source_index": index} for index in range(5)],
        }
        config["config_fingerprint"] = nf.canonical_hash(config)
        (feature_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
        rows = synthetic_rows()
        write_csv(feature_dir / "speakers.csv", list(rows[0]), rows)
        metadata_sha256 = nf.sha256_file(feature_dir / "speakers.csv")
        metadata = {
            "status": "ok", "config_fingerprint": config["config_fingerprint"], "completed": completed,
            "utterance_shape": [5, 12, 768], "speaker_metadata_sha256": metadata_sha256,
        }
        (feature_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        np.savez_compressed(
            feature_dir / "speaker_features.npz", ssl=self.ssl, mfcc=self.mfcc,
            speaker_id=np.asarray([row["speaker_id"] for row in rows], dtype="U32"),
            source_fingerprint=np.asarray(config["config_fingerprint"], dtype="U64"),
            speaker_metadata_sha256=np.asarray(metadata_sha256, dtype="U64"),
        )
        return feature_dir

    def test_predictor_whitelists_exclude_labels_ids_and_language(self):
        forbidden = {"height_cm", "weight_kg", "bmi", "speaker_id", "language"}
        self.assertEqual(set(evaluation.MODEL_ORDER), set(evaluation.INPUT_WHITELISTS))
        for columns in evaluation.INPUT_WHITELISTS.values():
            self.assertFalse(forbidden & set(columns))
        matrices = evaluation.build_predictors(synthetic_rows(), self.ssl, self.mfcc, [0, 1])
        self.assertEqual(matrices["demographics"].shape, (2, 2))
        self.assertEqual(matrices["acoustic_demographics"].shape, (2, 28))
        self.assertEqual(matrices["ssl12_demographics"].shape, (2, 770))

    def test_poisoned_test_labels_cannot_change_alpha_or_validation_predictions(self):
        first_rows = synthetic_rows("not-a-number")
        second_rows = synthetic_rows("999999")
        first = evaluation.select_validation_models(first_rows, self.ssl, self.mfcc)
        second = evaluation.select_validation_models(second_rows, self.ssl, self.mfcc)
        for name in evaluation.MODEL_ORDER:
            self.assertEqual(first["results"][name]["chosen_alpha"],
                             second["results"][name]["chosen_alpha"])
            self.assertTrue(np.array_equal(first["predictions"][name],
                                           second["predictions"][name]))

    def test_metric_values_known_fixture(self):
        metrics = evaluation._metrics(np.asarray([1.0, 3.0]), np.asarray([2.0, 2.0]))
        self.assertEqual(metrics["mae"], 1.0)
        self.assertEqual(metrics["rmse"], 1.0)
        self.assertEqual(metrics["r2"], 0.0)
        self.assertEqual(metrics["n"], 2)

    def test_pilot_and_missing_coverage_are_rejected(self):
        with self.assertRaisesRegex(nf.FeatureError, "pilot"):
            evaluation.load_feature_bundle(self.make_bundle(limit=100))
        with self.assertRaisesRegex(nf.FeatureError, "coverage"):
            evaluation.load_feature_bundle(self.make_bundle(completed=4))

    def test_reordered_speaker_csv_is_rejected(self):
        feature_dir = self.make_bundle()
        rows = nf.read_csv(feature_dir / "speakers.csv")
        write_csv(feature_dir / "speakers.csv", list(rows[0]), list(reversed(rows)))
        with self.assertRaisesRegex(nf.FeatureError, "order"):
            evaluation.load_feature_bundle(feature_dir)

    def test_changed_speaker_metadata_is_rejected_before_labels_are_used(self):
        feature_dir = self.make_bundle()
        rows = nf.read_csv(feature_dir / "speakers.csv")
        rows[0]["age"] = "999"
        write_csv(feature_dir / "speakers.csv", list(rows[0]), rows)
        with self.assertRaisesRegex(nf.FeatureError, "metadata hash"):
            evaluation.load_feature_bundle(feature_dir)

    def test_first_batch_outputs_validation_only_and_refuses_overwrite(self):
        feature_dir = self.make_bundle()
        output = self.root / "evaluation"
        result = evaluation.run_evaluation(feature_dir, output)
        self.assertEqual(result["scope"], "validation_only")
        with (output / "validation_predictions.csv").open(newline="") as handle:
            predictions = list(csv.DictReader(handle))
        self.assertEqual(len(predictions), 2)
        self.assertEqual({row["split"] for row in predictions}, {"val"})
        self.assertFalse(any("test" in path.name for path in output.rglob("*")))
        with self.assertRaisesRegex(nf.FeatureError, "already exists"):
            evaluation.run_evaluation(feature_dir, output)


if __name__ == "__main__":
    unittest.main()
