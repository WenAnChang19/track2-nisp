import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv_partial_ridge as core  # noqa: E402
import evaluate_nisp_baselines_height as base  # noqa: E402
import evaluate_nisp_selection_permutation as permutation  # noqa: E402


class MappingAndBatchTests(unittest.TestCase):
    def test_mappings_are_deterministic_and_stay_within_sex(self):
        sex = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
        config = {"permutations": {"n_random_per_target": 4, "seed": 2718}}
        first, first_hashes = permutation.build_mappings(sex, config)
        second, second_hashes = permutation.build_mappings(sex, config)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(first_hashes, second_hashes)
        self.assertTrue(np.array_equal(first[:, 0], np.tile(np.arange(8), (2, 1))))
        for mapping in first.reshape(-1, len(sex)):
            self.assertTrue(np.array_equal(sex[mapping], sex))
            self.assertEqual(sorted(mapping.tolist()), list(range(len(sex))))

    def test_batched_targets_match_one_column_at_a_time(self):
        rng = np.random.default_rng(19)
        n = 12
        sex = np.tile([0, 1], n // 2)
        age = np.arange(n, dtype=float) + 20
        data = {
            "sex": sex,
            "c": np.column_stack((np.ones(n), age, sex)),
            "targets": {
                "bmi": 18 + age * .11 + rng.normal(size=n),
                "height_cm": 145 + age * .4 + rng.normal(size=n),
            },
        }
        ssl = rng.normal(size=(n, 12, 5))
        folds = [
            {"train": np.arange(0, 8), "validation": np.arange(8, 12)},
            {"train": np.arange(4, 12), "validation": np.arange(0, 4)},
        ]
        config = {
            "targets": ["bmi", "height_cm"], "layers": list(range(1, 13)),
            "alpha_grid": [.1, 1e12],
            "permutations": {"n_random_per_target": 2, "seed": 31},
            "nesting_check": {"alpha": 1e12,
                              "prediction_absolute_tolerance": 1e-6,
                              "mae_absolute_tolerance": 1e-6},
        }
        mappings, _ = permutation.build_mappings(sex, config)
        _, columns, search, baseline, nesting = permutation.evaluate_cube(
            data, ssl, np.arange(n), folds, mappings, config)
        self.assertEqual(search.shape, (2, 3, 12, 2, 2))
        self.assertEqual(baseline.shape, (2, 3, 2))
        self.assertEqual(len(nesting), 48)

        for layer in config["layers"]:
            x = ssl[:, layer - 1]
            for fold_number, fold in enumerate(folds):
                train, validation = fold["train"], fold["validation"]
                cache = core.prepare_partial_fold(
                    data["c"][train], data["c"][validation], x[train], x[validation])
                for target in range(2):
                    for column in range(3):
                        y = columns[target, :, column]
                        prediction, demographic = cache.predict_all(y[train], config["alpha_grid"])
                        expected = np.mean(
                            np.abs(prediction[:, :, 0] - y[validation, None]), axis=0)
                        self.assertTrue(np.allclose(
                            search[target, column, layer - 1, fold_number], expected,
                            atol=1e-11, rtol=1e-11))
                        expected_baseline = np.mean(np.abs(demographic[:, 0] - y[validation]))
                        self.assertAlmostEqual(
                            baseline[target, column, fold_number], expected_baseline, places=11)


class SelectionAndGateTests(unittest.TestCase):
    def selection_fixture(self):
        search = np.full((2, 2, 12, 2, 2), 3.0)
        baseline = np.full((2, 2, 2), 4.0)
        # BMI observed: raw layer 1, but near-tie preference chooses layer 5;
        # layer 5 independently selects the second alpha.
        search[0, 0, 0, :, 0] = 1.0
        search[0, 0, 4, :, :] = [1.04, 1.03]
        # BMI random column fully reruns selection and chooses layer 6.
        search[0, 1, 9, :, 0] = 1.0
        search[0, 1, 5, :, 0] = 1.04
        # Height has its own band and alpha choices.
        search[1, 0, 1, :, 1] = 1.0
        search[1, 0, 5, :, 0] = 1.09
        search[1, 1, 11, :, 0] = .9
        config = {
            "targets": ["bmi", "height_cm"], "layers": list(range(1, 13)),
            "alpha_grid": [.1, 10.0], "alpha_selection": {"tie_tolerance": 1e-6},
            "layer_selection": {
                "near_tie_absolute_mae_band": {"bmi": .05, "height_cm": .1},
                "preference_layer": 6,
            },
        }
        hashes = [[f"t{target}p{column}" for column in range(2)] for target in range(2)]
        return search, baseline, hashes, config

    def test_each_permutation_reselects_layer_and_target_layer_alpha(self):
        rows, details = permutation.select_all(*self.selection_fixture())
        keyed = {(row["target"], row["permutation"]): row for row in rows}
        self.assertEqual(keyed[("bmi", 0)]["chosen_layer"], 5)
        self.assertEqual(keyed[("bmi", 0)]["chosen_alpha"], 10.0)
        self.assertEqual(keyed[("bmi", 1)]["chosen_layer"], 6)
        self.assertEqual(keyed[("height_cm", 0)]["chosen_layer"], 6)
        self.assertEqual(details[1][0]["selected_alphas_by_layer"][2], 10.0)

    def test_observed_bridge_checks_layer_scores_demographic_and_gain(self):
        search, baseline, hashes, config = self.selection_fixture()
        rows, details = permutation.select_all(search, baseline, hashes, config)
        for target_index in range(2):
            details[target_index][0]["layer_scores"] = {
                layer: float(search[target_index, 0, layer - 1, :,
                                    list(config["alpha_grid"]).index(
                                        details[target_index][0]["selected_alphas_by_layer"][layer])].mean())
                for layer in config["layers"]
            }
        config["observed_reproduction"] = {"mae_absolute_tolerance": 1e-12}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = {"layers_by_target": {}}
            selections = {"selection_by_target": {}}
            for target_index, target in enumerate(config["targets"]):
                metrics["layers_by_target"][target] = {
                    str(layer): {
                        "selected_alpha": details[target_index][0]["selected_alphas_by_layer"][layer],
                        "mean_fold_mae": details[target_index][0]["layer_scores"][layer],
                        "mean_demographics_fold_mae": 4.0,
                    } for layer in config["layers"]
                }
                selections["selection_by_target"][target] = {
                    "selected_layer": details[target_index][0]["selected_layer"],
                    "selected_alpha": details[target_index][0]["selected_alpha"],
                    "selected_score": next(row["selected_mae"] for row in rows
                                           if row["target"] == target and row["permutation"] == 0),
                }
            (root / "layer_cv_metrics.json").write_text(json.dumps(metrics))
            (root / "layer_selection.json").write_text(json.dumps(selections))
            self.assertTrue(permutation.reproduce_observed(rows, details, root, config)["passed"])
            rows[0]["demographics_mae"] += .01
            self.assertFalse(permutation.reproduce_observed(rows, details, root, config)["passed"])

    def minimal_run(self, output, nesting_passed=True, bridge_passed=True):
        config = {
            "scope": "fixture", "development_splits": ["train", "val"],
            "targets": ["bmi", "height_cm"], "layers": list(range(1, 13)),
            "alpha_grid": [1e12], "permutations": {"n_random_per_target": 0, "seed": 4},
            "nesting_check": {"records": 1},
        }
        rows = [{"split": "train"}]
        data = {"ids": ["S"], "sex": np.asarray([0]), "heldout_count": 1,
                "targets": {"bmi": np.asarray([1.]), "height_cm": np.asarray([2.])}}
        nesting = [{"passed": nesting_passed, "label_columns_checked": 1}]
        cube = (np.zeros((2, 1)), np.zeros((2, 1, 1)),
                np.zeros((2, 1, 12, 1, 1)), np.zeros((2, 1, 1)), nesting)
        mocked_rows = [{"target": "bmi", "permutation": 0}]
        mocked_details = [[{"selected_alphas_by_layer": {
            layer: 1e12 for layer in range(1, 13)}}] for _ in range(2)]
        stack = [
            patch.object(permutation, "load_config", return_value=(config, b"{}")),
            patch.object(permutation.evaluate_nisp, "load_feature_bundle",
                         return_value=(rows, np.zeros((1, 12, 1)), np.zeros((1, 1)), "fp")),
            patch.object(permutation.base, "load_development", return_value=data),
            patch.object(permutation, "validate_references", return_value=([{}], {})),
            patch.object(permutation, "protocol_lock", return_value=({}, "lock")),
            patch.object(permutation, "evaluate_cube", return_value=cube),
            patch.object(permutation, "select_all", return_value=(mocked_rows, mocked_details)),
            patch.object(permutation, "reproduce_observed",
                         return_value={"passed": bridge_passed, "checks": []}),
        ]
        with stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack[6], stack[7]:
            return permutation.run(Path("f"), Path("c"), Path("l"), Path("b"), Path("p"), output)

    def test_nesting_failure_stops_before_diagnostic_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            with self.assertRaisesRegex(permutation.PermutationError, "nesting"):
                self.minimal_run(output, nesting_passed=False)
            self.assertFalse((output / "selection_permutation_summary.json").exists())

    def test_observed_bridge_failure_stops_before_diagnostic_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            with self.assertRaisesRegex(permutation.PermutationError, "reproduce"):
                self.minimal_run(output, bridge_passed=False)
            self.assertFalse((output / "selection_permutation_summary.json").exists())

    def test_heldout_labels_and_demographics_are_never_parsed(self):
        class ProtectedTestRow(dict):
            def __getitem__(self, key):
                if key != "split":
                    raise AssertionError(f"protected test field read: {key}")
                return "test"

        rows = [
            {"split": "train", "speaker_id": "A", "sex": "Female", "age": "20",
             "bmi": "20", "height_cm": "160"},
            {"split": "val", "speaker_id": "B", "sex": "Male", "age": "30",
             "bmi": "25", "height_cm": "175"},
            ProtectedTestRow(split="test"),
        ]
        config = {"development_splits": ["train", "val"],
                  "expected_development_speakers": 2, "expected_heldout_speakers": 1}
        data = base.load_development(rows, np.zeros((3, 12, 2)), np.zeros((3, 2)), config)
        self.assertEqual(data["ids"], ["A", "B"])
        self.assertEqual(data["heldout_count"], 1)


if __name__ == "__main__":
    unittest.main()
