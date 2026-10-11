import sys
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv_partial_ridge as core  # noqa: E402
import evaluate_nisp_test as official  # noqa: E402
import nisp_features as nf  # noqa: E402
import nisp_final_inputs as final_inputs  # noqa: E402


class InputIsolationTests(unittest.TestCase):
    def config(self, development=2, test=1):
        return {
            "development_splits": ["train", "val"],
            "expected_development_speakers": development,
            "expected_test_speakers": test,
            "targets": ["bmi", "height_cm"],
            "group_analysis": {"categories": {
                "sex": ["Female", "Male"],
                "language": ["Hin", "Kan", "Mal", "Tam", "Tel"],
            }},
        }

    def test_projected_test_row_never_reads_target_or_weight_fields(self):
        class ProtectedRow(dict):
            def __getitem__(self, key):
                if key in {"bmi", "height_cm", "weight_kg"}:
                    raise AssertionError(f"protected test field accessed: {key}")
                return super().__getitem__(key)

        rows = [
            {"speaker_id": "A", "split": "train", "age": "20", "sex": "Female",
             "language": "Hin", "bmi": "21", "height_cm": "160"},
            ProtectedRow(speaker_id="T", split="test", age="30", sex="Male",
                         language="Kan", bmi="poison", height_cm="poison", weight_kg="poison"),
        ]
        predictors, targets = final_inputs.project_rows(rows, self.config(1, 1))
        self.assertEqual(tuple(predictors[0]), final_inputs.PREDICTOR_FIELDS)
        self.assertEqual([row["speaker_id"] for row in targets], ["A"])
        self.assertNotIn("bmi", predictors[1])
        self.assertNotIn("height_cm", predictors[1])
        self.assertNotIn("weight_kg", predictors[1])

    def make_prepared(self, root, feature_ids=None):
        predictors = [
            {"speaker_id": "A", "split": "train", "age": "20", "sex": "Female",
             "language": "Hin"},
            {"speaker_id": "B", "split": "val", "age": "30", "sex": "Male",
             "language": "Kan"},
            {"speaker_id": "T", "split": "test", "age": "40", "sex": "Female",
             "language": "Hin"},
        ]
        targets = [
            {"speaker_id": "A", "split": "train", "bmi": "20", "height_cm": "160"},
            {"speaker_id": "B", "split": "val", "bmi": "25", "height_cm": "175"},
        ]
        final_inputs.write_csv(root / "predictors.csv", final_inputs.PREDICTOR_FIELDS, predictors)
        final_inputs.write_csv(root / "development_targets.csv",
                               final_inputs.DEVELOPMENT_TARGET_FIELDS, targets)
        locked = b"fixture locked config\n"
        (root / "locked_config.json").write_bytes(locked)
        extraction_fp = "fixture-extraction-fingerprint"
        source_csv_sha = "a" * 64
        ids = feature_ids or ["A", "B", "T"]
        np.savez_compressed(
            root / "validated_features.npz",
            ssl=np.zeros((3, 12, 768), dtype=np.float32),
            mfcc=np.zeros((3, 26), dtype=np.float32),
            speaker_id=np.asarray(ids, dtype="U32"),
            source_fingerprint=np.asarray(extraction_fp, dtype="U64"),
            speaker_metadata_sha256=np.asarray(source_csv_sha, dtype="U64"),
        )
        npz_sha = nf.sha256_file(root / "validated_features.npz")
        source_fp = nf.canonical_hash({"config_fingerprint": extraction_fp,
                                       "speaker_csv": source_csv_sha,
                                       "speaker_npz": npz_sha})
        prepared = {name: nf.sha256_file(root / name) for name in
                    ("predictors.csv", "development_targets.csv", "validated_features.npz",
                     "locked_config.json")}
        manifest = {
            "status": "ok", "raw_config_sha256": final_inputs.sha256_bytes(locked),
            "feature_source_fingerprint": source_fp,
            "feature_extraction_config_fingerprint": extraction_fp,
            "source_evidence_sha256": {"speakers.csv": source_csv_sha,
                                       "speaker_features.npz": npz_sha},
            "prepared_file_sha256": prepared, "test_target_fields_accessed": False,
            "speaker_order_hash": nf.canonical_hash(["A", "B", "T"]),
            "development_ids_hash": nf.canonical_hash(["A", "B"]),
            "test_ids_hash": nf.canonical_hash(["T"]),
            "counts": {"all": 3, "development": 2, "test": 1},
        }
        final_inputs.write_json(root / "view_manifest.json", manifest)
        config = self.config()
        config["expected_feature_source_fingerprint"] = source_fp
        return config, final_inputs.sha256_bytes(locked)

    def test_validated_npz_id_order_must_match_sanitized_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, lock_sha = self.make_prepared(root, ["A", "T", "B"])
            with patch.object(final_inputs, "CONFIG_SHA256", lock_sha):
                with self.assertRaisesRegex(final_inputs.FinalInputError, "ID order"):
                    final_inputs.load_prepared_inputs(root, config)


class PredictionMathTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(13)
        n, n_dev = 9, 6
        rows = []
        for index in range(n):
            rows.append({"speaker_id": f"S{index}",
                         "split": "train" if index < n_dev else "test",
                         "language": "Hin", "sex": "Female" if index % 2 == 0 else "Male"})
        age = np.arange(n, dtype=float) + 20
        sex = np.asarray([index % 2 for index in range(n)], dtype=float)
        ssl = rng.normal(size=(n, 12, 5))
        mfcc = rng.normal(size=(n, 4))
        targets = {"bmi": 18 + age[:n_dev] * .2 + rng.normal(size=n_dev),
                   "height_cm": 145 + age[:n_dev] * .8 + rng.normal(size=n_dev)}
        data = {"rows": rows, "ids": [row["speaker_id"] for row in rows],
                "age": age, "sex": sex, "ssl": ssl, "mfcc": mfcc, "targets": targets,
                "dev_ids": [f"S{i}" for i in range(n_dev)],
                "test_ids": [f"S{i}" for i in range(n_dev, n)]}
        config = {"development_splits": ["train", "val"],
                  "targets": ["bmi", "height_cm"],
                  "model_order": list(official.MODELS),
                  "settings_by_target": {
                      "bmi": {"selected_layer": 5, "selected_ssl_alpha": 2.0,
                              "mfcc_alpha": 3.0, "ssl12_alpha": 4.0},
                      "height_cm": {"selected_layer": 3, "selected_ssl_alpha": 5.0,
                                    "mfcc_alpha": 6.0, "ssl12_alpha": 7.0},
                  }}
        return data, config

    def test_full_development_partial_models_match_augmented_oracle(self):
        data, config = self.fixture()
        test_indices, predictions = official.fit_predictions(data, config)
        dev, test, c = official.build_design(data, config)
        self.assertTrue(np.array_equal(test_indices, test))
        for target in config["targets"]:
            settings = config["settings_by_target"][target]
            matrices = {
                "mfcc_demographics": (data["mfcc"], settings["mfcc_alpha"]),
                "selected_ssl_demographics": (
                    data["ssl"][:, settings["selected_layer"] - 1],
                    settings["selected_ssl_alpha"]),
                "ssl12_demographics": (data["ssl"][:, 11], settings["ssl12_alpha"]),
            }
            for model, (x, alpha) in matrices.items():
                expected = core.direct_partial_penalty_prediction(
                    c[dev], c[test], x[dev], x[test], data["targets"][target], alpha)
                self.assertTrue(np.allclose(predictions[target][model], expected,
                                            atol=1e-9, rtol=1e-9))

    def test_poisoned_test_target_columns_cannot_change_predictions(self):
        data, config = self.fixture()
        first = official.fit_predictions(data, config)[1]
        for row in data["rows"]:
            if row["split"] == "test":
                row["bmi"] = "poison-a"; row["height_cm"] = "poison-b"
        second = official.fit_predictions(data, config)[1]
        for target in config["targets"]:
            for model in official.MODELS:
                self.assertTrue(np.array_equal(first[target][model], second[target][model]))

    def test_prediction_rows_are_target_major_and_use_exact_test_order(self):
        data, config = self.fixture()
        test, predictions = official.fit_predictions(data, config)
        rows, audit = official.prediction_rows_and_audit(data, test, predictions, config)
        self.assertEqual([(row["target"], row["speaker_id"]) for row in rows],
                         [(target, speaker) for target in config["targets"]
                          for speaker in data["test_ids"]])
        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["prediction_columns"], list(official.PREDICTION_FIELDS))


class ScoringTests(unittest.TestCase):
    def test_prediction_gate_failure_occurs_before_test_target_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "score"
            with patch.object(official, "validate_prediction_artifacts",
                              side_effect=official.OfficialTestError("prediction hash mismatch")), \
                    patch.object(official, "load_test_targets") as target_loader:
                with self.assertRaisesRegex(official.OfficialTestError, "hash mismatch"):
                    official.run_score(Path("prediction"), Path("config"),
                                       Path("protected-targets"), output)
                target_loader.assert_not_called()
            self.assertFalse(output.exists())

    def test_paired_bootstrap_hand_fixture_uses_the_supplied_same_indices(self):
        actual = np.asarray([0.0, 0.0])
        predictions = {"candidate": np.asarray([1.0, 3.0]),
                       "reference": np.asarray([2.0, 0.0])}
        indices = np.asarray([[0, 0], [1, 1], [0, 1]])
        distribution, point = official.paired_bootstrap(
            actual, predictions, indices, "candidate", "reference")
        self.assertTrue(np.array_equal(distribution, np.asarray([-1.0, 3.0, 1.0])))
        self.assertEqual(point, 1.0)

        config = {
            "targets": ["bmi", "height_cm"],
            "bootstrap": {"seed": 8, "n_resamples": 7,
                          "method": "paired", "uncertainty_scope": "fixture",
                          "individual_percentile_quantiles": [.025, .975],
                          "two_target_bonferroni_two_sided_percentile_quantiles": [.0125, .9875]},
            "primary_comparisons": [
                {"target": "bmi", "candidate": "candidate", "reference": "reference"},
                {"target": "height_cm", "candidate": "candidate", "reference": "reference"},
            ],
            "secondary_comparisons": [],
        }
        actuals = {target: actual for target in config["targets"]}
        all_predictions = {target: predictions for target in config["targets"]}
        shared, records, distributions = official.bootstrap_outputs(
            actuals, all_predictions, config)
        self.assertEqual(shared.shape, (7, 2))
        self.assertTrue(np.array_equal(distributions[0], distributions[1]))
        self.assertEqual(len(records), 2)

    def test_all_fixed_group_categories_targets_and_models_are_reported(self):
        config = {
            "targets": ["bmi", "height_cm"], "model_order": list(official.MODELS),
            "group_analysis": {"fields": ["sex", "language"], "categories": {
                "sex": ["Female", "Male"],
                "language": ["Hin", "Kan", "Mal", "Tam", "Tel"],
            }},
        }
        data = {"rows": [
            {"split": "test", "sex": "Female", "language": "Hin"},
            {"split": "test", "sex": "Male", "language": "Kan"},
        ]}
        actuals = {target: np.asarray([1.0, 2.0]) for target in config["targets"]}
        predictions = {target: {model: np.asarray([1.0, 2.0])
                                for model in official.MODELS}
                       for target in config["targets"]}
        rows, csv_rows = official.group_outputs(actuals, predictions, data, config)
        self.assertEqual(len(rows), 7 * 2 * 5)
        self.assertEqual(len(csv_rows), len(rows))
        zero = [row for row in rows if row["group_field"] == "language" and
                row["group_value"] == "Tel"]
        self.assertEqual(len(zero), 2 * 5)
        self.assertTrue(all(row["n"] == 0 and row["mae"] is None for row in zero))

    def test_toy_full_prepare_predict_score_artifact_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_npz = root / "source_features.npz"
            target_source = root / "source_speakers.csv"
            n_dev, n_test = 210, 135
            all_ids = [f"D{i:03d}" for i in range(n_dev)] + [f"T{i:03d}" for i in range(n_test)]
            rows = []
            languages = ["Hin", "Kan", "Mal", "Tam"]  # Tel intentionally has zero test rows.
            for index, speaker_id in enumerate(all_ids):
                is_test = index >= n_dev
                split = "test" if is_test else ("train" if index < 168 else "val")
                rows.append({"speaker_id": speaker_id, "split": split,
                             "age": str(20 + index % 40),
                             "sex": "Female" if index % 2 == 0 else "Male",
                             "language": languages[index % len(languages)],
                             "bmi": str(19 + (index % 17) * .3),
                             "height_cm": str(150 + index % 31)})
            with target_source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle,
                                        fieldnames=("speaker_id", "split", "bmi", "height_cm"),
                                        lineterminator="\n")
                writer.writeheader()
                writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
            source_csv_sha = nf.sha256_file(target_source)
            extraction_fp = "toy-extraction"
            np.savez_compressed(
                source_npz, ssl=np.zeros((345, 12, 2), dtype=np.float32),
                mfcc=np.zeros((345, 2), dtype=np.float32),
                speaker_id=np.asarray(all_ids, dtype="U32"),
                source_fingerprint=np.asarray(extraction_fp, dtype="U64"),
                speaker_metadata_sha256=np.asarray(source_csv_sha, dtype="U64"),
            )
            npz_sha = nf.sha256_file(source_npz)
            source_fp = nf.canonical_hash({"config_fingerprint": extraction_fp,
                                           "speaker_csv": source_csv_sha,
                                           "speaker_npz": npz_sha})
            raw = b"toy official config\n"
            raw_sha = final_inputs.sha256_bytes(raw)
            config = {
                "development_splits": ["train", "val"],
                "expected_development_speakers": n_dev, "expected_test_speakers": n_test,
                "expected_feature_source_fingerprint": source_fp,
                "targets": ["bmi", "height_cm"], "model_order": list(official.MODELS),
                "settings_by_target": {
                    "bmi": {"selected_layer": 5, "selected_ssl_alpha": 1000.,
                            "mfcc_alpha": 1e12, "ssl12_alpha": 1000.},
                    "height_cm": {"selected_layer": 3, "selected_ssl_alpha": 1000.,
                                  "mfcc_alpha": 100., "ssl12_alpha": 1000.},
                },
                "bootstrap": {"seed": 9, "n_resamples": 11,
                              "method": "paired percentile", "uncertainty_scope": "toy",
                              "individual_percentile_quantiles": [.025, .975],
                              "two_target_bonferroni_two_sided_percentile_quantiles":
                              [.0125, .9875]},
                "primary_comparisons": [
                    {"target": target, "candidate": "selected_ssl_demographics",
                     "reference": "demographics"} for target in ("bmi", "height_cm")],
                "secondary_comparisons": [
                    {"candidate": "selected_ssl_demographics", "reference": "mfcc_demographics"},
                    {"candidate": "ssl12_demographics", "reference": "demographics"},
                    {"candidate": "selected_ssl_demographics", "reference": "ssl12_demographics"},
                ],
                "group_analysis": {"fields": ["sex", "language"],
                                   "categories": {"sex": ["Female", "Male"],
                                                  "language": ["Hin", "Kan", "Mal", "Tam", "Tel"]},
                                   "small_group_note": "toy"},
                "scientific_wording": "toy association only",
            }
            evidence = {"metadata.json": "m", "config.json": "c",
                        "speakers.csv": source_csv_sha, "speaker_features.npz": npz_sha}
            inputs = root / "inputs"; prediction_dir = root / "predictions"; score_dir = root / "score"

            test_indices = np.arange(n_dev, n_dev + n_test)
            fake_predictions = {}
            for target in config["targets"]:
                base_values = np.linspace(20, 30, n_test) if target == "bmi" else \
                    np.linspace(155, 175, n_test)
                fake_predictions[target] = {
                    model: base_values + offset * .01
                    for offset, model in enumerate(official.MODELS)
                }
            stable_sources = {"evaluate_nisp_test.py": "source-hash",
                              "nisp_final_inputs.py": "input-source-hash"}
            with patch.object(nf, "SSL_SHAPE", (12, 2)), \
                    patch.object(nf, "MFCC_SIZE", 2), \
                    patch.object(final_inputs, "CONFIG_SHA256", raw_sha), \
                    patch.object(final_inputs, "load_config", return_value=(config, raw)), \
                    patch.object(final_inputs, "validate_source_bundle",
                                 return_value=(rows, source_npz, source_fp, evidence, extraction_fp)):
                prepared = final_inputs.prepare(Path("source"), Path("config"), inputs)
                self.assertFalse(prepared["test_target_fields_accessed"])
                with patch.object(official, "validate_references", return_value={}), \
                        patch.object(official, "source_hashes", return_value=stable_sources), \
                        patch.object(official, "fit_predictions",
                                     return_value=(test_indices, fake_predictions)):
                    predicted = official.run_predict(
                        inputs, Path("config"), root / "layer", root / "baseline",
                        root / "prior", prediction_dir)
                self.assertEqual(predicted["prediction_rows"], 270)
                with patch.object(official, "source_hashes", return_value=stable_sources):
                    scored = official.run_score(prediction_dir, Path("config"),
                                                target_source, score_dir)

            self.assertEqual(scored, {"status": "ok", "test_speakers": 135,
                                      "metric_rows": 10, "bootstrap_comparisons": 8,
                                      "group_rows": 70})
            predictions = official.read_exact_csv(
                prediction_dir / "test_predictions.csv", official.PREDICTION_FIELDS)
            self.assertEqual(len(predictions), 270)
            self.assertEqual([row["speaker_id"] for row in predictions[:135]], all_ids[210:])
            self.assertEqual([row["speaker_id"] for row in predictions[135:]], all_ids[210:])
            metrics = json.loads((score_dir / "test_metrics.json").read_text())
            self.assertEqual(sum(len(value) for value in
                                 metrics["metrics_by_target_and_model"].values()), 10)
            bootstrap = json.loads((score_dir / "bootstrap_comparisons.json").read_text())
            self.assertEqual(len(bootstrap["comparisons"]), 8)
            groups = json.loads((score_dir / "group_metrics.json").read_text())["rows"]
            self.assertEqual(len(groups), 70)
            self.assertEqual(len([row for row in groups if row["group_value"] == "Tel"]), 10)
            self.assertTrue(all(row["n"] == 0 for row in groups if row["group_value"] == "Tel"))


if __name__ == "__main__":
    unittest.main()
