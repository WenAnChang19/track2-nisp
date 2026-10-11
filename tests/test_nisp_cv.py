import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv_partial_ridge as core  # noqa: E402
import evaluate_nisp_cv as cv  # noqa: E402


class PartialRidgeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        n_train, n_val, width = 40, 11, 6
        age_train = self.rng.normal(30, 6, n_train)
        sex_train = np.tile([0.0, 1.0], n_train // 2)
        age_val = self.rng.normal(30, 6, n_val)
        sex_val = np.asarray(([0.0, 1.0] * 6)[:n_val])
        self.c_train = np.column_stack((np.ones(n_train), age_train, sex_train))
        self.c_val = np.column_stack((np.ones(n_val), age_val, sex_val))
        # Deliberately correlate audio with demographics to exercise both residualizations.
        self.x_train = (age_train[:, None] * 0.4 + sex_train[:, None] * 2.0 +
                        self.rng.normal(size=(n_train, width)))
        self.x_val = (age_val[:, None] * 0.4 + sex_val[:, None] * 2.0 +
                      self.rng.normal(size=(n_val, width)))
        self.y_train = 12 + 0.3 * age_train + 1.5 * sex_train + self.x_train[:, 0] * 0.7

    def test_svd_solver_matches_independent_augmented_partial_penalty(self):
        prepared = core.prepare_partial_fold(self.c_train, self.c_val,
                                             self.x_train, self.x_val)
        for alpha in (0.01, 3.0, 1000.0):
            predicted, _ = prepared.predict_all(self.y_train, [alpha])
            oracle = core.direct_partial_penalty_prediction(
                self.c_train, self.c_val, self.x_train, self.x_val, self.y_train, alpha
            )
            self.assertTrue(np.allclose(predicted[:, 0, 0], oracle, rtol=1e-9, atol=1e-9))

    def test_huge_alpha_degenerates_to_demographics(self):
        prepared = core.prepare_partial_fold(self.c_train, self.c_val,
                                             self.x_train, self.x_val)
        predicted, demographic = prepared.predict_all(self.y_train, [1e12])
        self.assertLess(np.max(np.abs(predicted[:, 0, 0] - demographic[:, 0])), 1e-6)

    def test_scaler_and_coefficients_use_training_only_and_predictions_ignore_heldout_y(self):
        prepared = core.prepare_partial_fold(self.c_train, self.c_val,
                                             self.x_train, self.x_val + 1e9)
        self.assertTrue(np.allclose(prepared.mean, self.x_train.mean(axis=0)))
        first, _ = prepared.predict_all(self.y_train, [1.0, 10.0])
        # There is no validation target argument; changing held-out labels cannot affect prediction.
        second, _ = prepared.predict_all(self.y_train.copy(), [1.0, 10.0])
        self.assertTrue(np.array_equal(first, second))

    def test_alpha_tie_chooses_largest_within_tolerance(self):
        chosen = core.choose_alpha({0.1: 1.0, 1.0: 1.0000005, 10.0: 1.000002}, 1e-6)
        self.assertEqual(chosen, 1.0)

    def test_shape_and_finite_guards_prevent_metric_broadcasting(self):
        with self.assertRaisesRegex(core.CVError, "same-shape"):
            core.regression_metrics(np.ones(3), np.ones((3, 1)))
        with self.assertRaisesRegex(core.CVError, "same-shape"):
            core.regression_metrics(np.ones(3), np.asarray([1.0, np.nan, 1.0]))
        prepared = core.prepare_partial_fold(self.c_train, self.c_val,
                                             self.x_train, self.x_val)
        with self.assertRaisesRegex(core.CVError, "alphas"):
            prepared.predict_all(self.y_train, [])
        with self.assertRaisesRegex(core.CVError, "tolerance"):
            core.choose_alpha({1.0: 2.0}, -1.0)

    def test_stratified_permutation_preserves_each_sex_multiset(self):
        target = np.asarray([1, 2, 3, 10, 20, 30], dtype=float)
        sex = np.asarray([0, 0, 0, 1, 1, 1])
        permuted, mapping, digest = core.stratified_permutation(
            target, sex, np.random.default_rng(43)
        )
        self.assertEqual(sorted(permuted[sex == 0]), sorted(target[sex == 0]))
        self.assertEqual(sorted(permuted[sex == 1]), sorted(target[sex == 1]))
        self.assertTrue(np.all(sex[mapping] == sex))
        self.assertEqual(len(digest), 64)


class CVProtocolTests(unittest.TestCase):
    def small_data(self):
        count = 20
        sex = np.tile([0, 1], count // 2)
        age = np.arange(count, dtype=float) + 20
        return {
            "ids": [f"S{index:02d}" for index in range(count)], "sex": sex,
            "age": age, "target": 20 + age * 0.1,
            "c": np.column_stack((np.ones(count), age, sex)),
            "audio": {
                "acoustic_demographics": np.column_stack((age, sex, np.sin(age))),
                "ssl12_demographics": np.column_stack((age ** 2, sex, np.cos(age))),
            },
            "source_splits": ["train"] * count,
        }

    def config(self):
        return {"cv": {"n_splits": 5, "n_repeats": 2, "random_state": 42}}

    def test_common_folds_are_reproducible_and_svd_is_cached_across_targets(self):
        data = self.small_data()
        first = cv.make_folds(data, self.config())
        second = cv.make_folds(data, self.config())
        self.assertEqual(
            [(f["train"].tolist(), f["validation"].tolist()) for f in first],
            [(f["train"].tolist(), f["validation"].tolist()) for f in second],
        )
        original = np.linalg.svd
        calls = []
        with patch("numpy.linalg.svd", side_effect=lambda *args, **kwargs:
                   (calls.append(1) or original(*args, **kwargs))):
            prepared = cv.prepare_folds(data, first)
            expected = len(first) * len(cv.AUDIO_MODELS)
            self.assertEqual(len(calls), expected)

    def test_folds_reject_a_sex_class_smaller_than_number_of_splits(self):
        data = self.small_data()
        data["sex"] = np.asarray([0] * 17 + [1] * 3)
        with self.assertRaisesRegex(core.CVError, "at least n_splits"):
            cv.make_folds(data, self.config())
            cv.score_target(prepared, data["target"], [0.1, 1.0])
            cv.score_target(prepared, data["target"][::-1], [0.1, 1.0])
            self.assertEqual(len(calls), expected)

    def test_prediction_schema_has_exactly_five_oof_rows_per_person_for_five_repeats(self):
        data = self.small_data()
        config = {"cv": {"n_splits": 5, "n_repeats": 5, "random_state": 42},
                  "alpha_grid": [1e12],
                  "paired_comparison": {"zero_tolerance": 1e-6}}
        folds = cv.make_folds(data, config)
        prepared = cv.prepare_folds(data, folds)
        scored = cv.score_target(prepared, data["target"], [1e12])
        chosen = {"demographics": None, "acoustic_demographics": 1e12,
                  "ssl12_demographics": 1e12}
        searches = {name: {1e12: 0.0} for name in cv.AUDIO_MODELS}
        _alpha, _fold, predictions, _summaries, _mae = cv._fold_outputs(
            data, prepared, scored, chosen, searches, config, "lock"
        )
        self.assertEqual(len(predictions), len(data["ids"]) * 5)
        self.assertEqual({row["speaker_id"] for row in predictions}, set(data["ids"]))
        self.assertTrue(all("model" not in row for row in predictions))
        self.assertTrue(all("demographics_prediction" in row and
                            "acoustic_demographics_prediction" in row and
                            "ssl12_demographics_prediction" in row for row in predictions))

    def test_poisoned_test_rows_do_not_change_development_data(self):
        rows = []
        for index in range(4):
            rows.append({"speaker_id": f"D{index}", "split": "train" if index < 2 else "val",
                         "age": str(20 + index), "sex": "Female" if index % 2 == 0 else "Male",
                         "bmi": str(22 + index)})
        rows.append({"speaker_id": "TEST", "split": "test", "age": "poison",
                     "sex": "poison", "bmi": "poison"})
        ssl = np.zeros((5, 12, 768), dtype=np.float32)
        mfcc = np.zeros((5, 26), dtype=np.float32)
        config = {"development_splits": ["train", "val"],
                  "expected_development_speakers": 4, "expected_heldout_speakers": 1}
        first = cv.development_data(rows, ssl, mfcc, config)
        rows[-1].update(age="999", sex="Male", bmi="999")
        second = cv.development_data(rows, ssl, mfcc, config)
        for field in ("age", "sex", "target", "c"):
            self.assertTrue(np.array_equal(first[field], second[field]))

    def test_cli_rejects_out_of_scope_final_test_option(self):
        with self.assertRaises(SystemExit):
            cv.main(["--features", "x", "--config", "y", "--final-test"])


if __name__ == "__main__":
    unittest.main()
