import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audio_ridge_cv as audio  # noqa: E402
import evaluate_nisp_baselines_height as runner  # noqa: E402
import nisp_features as nf  # noqa: E402


class AudioRidgeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.x_train = rng.normal(size=(35, 7))
        self.x_val = rng.normal(size=(9, 7))
        self.y_train = 20 + self.x_train[:, 0] * 2 - self.x_train[:, 2] + rng.normal(size=35)

    def test_matches_independent_sklearn_pipeline(self):
        cache = audio.prepare_audio_fold(self.x_train, self.x_val)
        for alpha in (0.01, 3.0, 1000.0):
            predicted = cache.predict_all(self.y_train, [alpha])[0][:, 0, 0]
            oracle = Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, fit_intercept=True, solver="svd")),
            ]).fit(self.x_train, self.y_train).predict(self.x_val)
            self.assertTrue(np.allclose(predicted, oracle, rtol=1e-9, atol=1e-9))

    def test_huge_alpha_degenerates_to_training_mean(self):
        predicted, target_mean = audio.prepare_audio_fold(
            self.x_train, self.x_val).predict_all(self.y_train, [1e12])
        self.assertLess(np.max(np.abs(predicted[:, 0, 0] - self.y_train.mean())), 1e-6)
        self.assertAlmostEqual(target_mean[0], self.y_train.mean())


class BaselineProtocolTests(unittest.TestCase):
    def toy_data(self):
        count = 10
        age = np.arange(count, dtype=float) + 20
        sex = np.tile([0, 1], 5)
        return {"ids": [f"S{i}" for i in range(count)], "source_splits": ["train"] * count,
                "sex": sex, "c": np.column_stack((np.ones(count), age, sex)),
                "targets": {"bmi": age * .2, "height_cm": 150 + age},
                "audio": {"mfcc": np.column_stack((age, np.sin(age))),
                          "ssl12": np.column_stack((age ** 2, np.cos(age)))},
                "heldout_count": 1}

    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(runner.BaselineError, "refusing overwrite"):
                runner.require_new_output(Path(temporary))

    def test_prepare_reuses_partial_core_for_both_combined_models_and_svd_cache(self):
        data = self.toy_data()
        folds = [{"number": 0, "repeat": 0, "fold": 0,
                  "train": np.arange(5, 10), "validation": np.arange(0, 5)}]
        import cv_partial_ridge
        original = cv_partial_ridge.prepare_partial_fold
        with patch.object(cv_partial_ridge, "prepare_partial_fold", wraps=original) as combined:
            prepared = runner.prepare_folds(data, folds)
        self.assertEqual(combined.call_count, 2)
        first = runner.score_target(prepared, data["targets"]["bmi"], [1.0])
        second = runner.score_target(prepared, data["targets"]["height_cm"], [1.0])
        self.assertEqual(len(first["acoustic_only"]), 1)
        self.assertEqual(len(second["ssl12_demographics"]), 1)

    def test_constants_use_training_target_only(self):
        data = self.toy_data()
        folds = [{"number": 0, "repeat": 0, "fold": 0,
                  "train": np.arange(5, 10), "validation": np.arange(0, 5)}]
        prepared = runner.prepare_folds(data, folds)
        first = runner.score_target(prepared, data["targets"]["bmi"], [1.0])
        poisoned = data["targets"]["bmi"].copy(); poisoned[:5] = 999999
        second = runner.score_target(prepared, poisoned, [1.0])
        for name in ("mean", "median"):
            self.assertTrue(np.array_equal(first[name][0]["prediction"],
                                           second[name][0]["prediction"]))

    def test_poisoned_test_fields_are_never_parsed(self):
        rows = [{"speaker_id": "D0", "split": "train", "age": "20", "sex": "Female",
                 "bmi": "21", "height_cm": "160"},
                {"speaker_id": "D1", "split": "val", "age": "30", "sex": "Male",
                 "bmi": "25", "height_cm": "180"},
                {"speaker_id": "T", "split": "test", "age": "poison", "sex": "poison",
                 "bmi": "poison", "height_cm": "poison"}]
        ssl = np.zeros((3, 12, 768), dtype=np.float32); mfcc = np.zeros((3, 26), dtype=np.float32)
        config = {"development_splits": ["train", "val"],
                  "expected_development_speakers": 2, "expected_heldout_speakers": 1}
        first = runner.load_development(rows, ssl, mfcc, config)
        rows[-1].update(age="1", sex="Male", bmi="1", height_cm="1")
        second = runner.load_development(rows, ssl, mfcc, config)
        for target in ("bmi", "height_cm"):
            self.assertTrue(np.array_equal(first["targets"][target], second["targets"][target]))

    def test_each_target_selects_its_own_alpha(self):
        def fake(best_index):
            scored = {name: [] for name in runner.MODELS}
            for name in runner.MODELS:
                if name in runner.AUDIO_MODELS:
                    scored[name] = [{"metrics": [{"mae": 1.0 if i == best_index else 2.0}
                                                   for i in range(2)]}]
                else:
                    scored[name] = [{"metrics": {"mae": 1.0}}]
            return scored
        bmi, _ = runner.select_models(fake(0), [0.1, 10.0], 1e-6)
        height, _ = runner.select_models(fake(1), [0.1, 10.0], 1e-6)
        self.assertEqual(bmi["ssl12_only"], 0.1)
        self.assertEqual(height["ssl12_only"], 10.0)

    def test_prior_fold_hash_ids_and_per_repeat_coverage(self):
        data = self.toy_data()
        folds = []
        for fold in range(2):
            validation = data["ids"][fold * 5:(fold + 1) * 5]
            train = [speaker for speaker in data["ids"] if speaker not in validation]
            folds.append({"repeat": 0, "fold": fold, "train_ids": train,
                          "validation_ids": validation})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = {"ordered_dev_ids_hash": nf.canonical_hash(data["ids"]),
                    "full_fold_train_and_validation_ids": folds,
                    "feature_source_fingerprint": "bundle"}
            (root / "protocol_lock.json").write_text(json.dumps(lock))
            document = {"protocol_lock_sha256": nf.sha256_file(root / "protocol_lock.json"),
                        "folds": folds}
            (root / "folds.json").write_text(json.dumps(document))
            config = {"prior_cv_reference": {
                "protocol_lock_sha256": nf.sha256_file(root / "protocol_lock.json"),
                "folds_json_sha256": nf.sha256_file(root / "folds.json"),
                "ordered_dev_ids_hash": nf.canonical_hash(data["ids"])},
                "expected_feature_source_fingerprint": "bundle",
                "cv": {"n_splits": 2, "n_repeats": 1}}
            loaded, _ = runner.load_prior_folds(root, data, config)
            self.assertEqual(len(loaded), 2)
            folds[0]["validation_ids"][0] = "UNKNOWN"
            (root / "folds.json").write_text(json.dumps({"protocol_lock_sha256": config["prior_cv_reference"]["protocol_lock_sha256"], "folds": folds}))
            config["prior_cv_reference"]["folds_json_sha256"] = nf.sha256_file(root / "folds.json")
            with self.assertRaisesRegex(runner.BaselineError, "lock and folds"):
                runner.load_prior_folds(root, data, config)

    def test_bmi_reproduction_is_keyed_not_row_order_and_can_fail_before_height(self):
        summaries = {name: {"selected_alpha": None if name == "demographics" else 1.0,
                            "mean_fold_mae": 2.0} for name in
                     ("demographics", "acoustic_demographics", "ssl12_demographics")}
        current = [{"speaker_id": "A", "repeat": 0, "fold": 0,
                    "demographics_prediction": 1.0, "acoustic_demographics_prediction": 2.0,
                    "ssl12_demographics_prediction": 3.0}]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cv_metrics.json").write_text(json.dumps({"models": summaries}))
            fields = ["speaker_id", "repeat", "fold", "demographics_prediction",
                      "acoustic_demographics_prediction", "ssl12_demographics_prediction"]
            import csv
            with (root / "cv_predictions.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fields); writer.writeheader(); writer.writerows(reversed(current))
            config = {"bmi_reproduction": {"prior_models": list(summaries),
                      "mae_absolute_tolerance": 1e-9, "prediction_absolute_tolerance": 1e-8}}
            self.assertTrue(runner.reproduce_prior(root, summaries, current, config)["passed"])
            current[0]["ssl12_demographics_prediction"] += 1
            self.assertFalse(runner.reproduce_prior(root, summaries, current, config)["passed"])


if __name__ == "__main__":
    unittest.main()
