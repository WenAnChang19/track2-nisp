import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv_partial_ridge as core  # noqa: E402
import evaluate_nisp_layers as layers  # noqa: E402


class LayerMathTests(unittest.TestCase):
    def test_layer_indexing_is_k_minus_one(self):
        ssl = np.zeros((3, 12, 2), dtype=np.float32)
        for index in range(12):
            ssl[:, index, :] = index + 1
        selected = layers.layer_matrix(ssl, np.asarray([0, 2]), 6)
        self.assertTrue(np.all(selected == 6))
        with self.assertRaises(layers.LayerError):
            layers.layer_matrix(ssl, np.asarray([0]), 0)

    def test_partial_solver_matches_independent_oracle(self):
        rng = np.random.default_rng(3)
        age = np.arange(30, dtype=float) + 20
        sex = np.tile([0.0, 1.0], 15)
        c = np.column_stack((np.ones(30), age, sex))
        x = age[:, None] * .2 + sex[:, None] + rng.normal(size=(30, 4))
        y = 10 + age * .3 + x[:, 0]
        cache = core.prepare_partial_fold(c[:22], c[22:], x[:22], x[22:])
        prediction = cache.predict_all(y[:22], [3.0])[0][:, 0, 0]
        oracle = core.direct_partial_penalty_prediction(c[:22], c[22:], x[:22], x[22:], y[:22], 3.0)
        self.assertTrue(np.allclose(prediction, oracle, atol=1e-9, rtol=1e-9))

    def test_target_layer_alpha_selection_is_independent(self):
        def scored(best):
            return [{"metrics": [{"mae": 1.0 if i == best else 2.0} for i in range(2)]}]
        bmi, _ = layers.select_alpha(scored(0), [0.1, 10.0], 1e-6)
        height, _ = layers.select_alpha(scored(1), [0.1, 10.0], 1e-6)
        self.assertEqual(bmi, 0.1); self.assertEqual(height, 10.0)

    def test_layer_band_prefers_six_then_closest_then_lower_and_reports_raw_min(self):
        scores = {layer: 2.0 for layer in range(1, 13)}
        scores[1] = 1.0; scores[5] = 1.05; scores[7] = 1.05
        result = layers.choose_layer(scores, .05, 6)
        self.assertEqual(result["raw_minimum_layer"], 1)
        self.assertEqual(result["selected_layer"], 5)
        self.assertIn(5, result["eligible_layers"])
        self.assertAlmostEqual(result["selected_score_minus_raw_minimum"], .05)
        scores[6] = 1.04
        self.assertEqual(layers.choose_layer(scores, .05, 6)["selected_layer"], 6)
        with self.assertRaises(layers.LayerError):
            layers.choose_layer(scores, -1)


class BridgeAndRunTests(unittest.TestCase):
    def write_reference(self, root):
        metrics = {"models_by_target": {}}
        rows = []
        for target in ("bmi", "height_cm"):
            metrics["models_by_target"][target] = {
                "ssl12_demographics": {"selected_alpha": 10.0, "mean_fold_mae": 2.0},
                "demographics": {"selected_alpha": None, "mean_fold_mae": 2.5},
            }
            rows.append({"target": target, "speaker_id": "S", "repeat": "0", "fold": "0",
                         "ssl12_demographics_prediction": "3.0", "demographics_prediction": "2.0"})
        (root / "cv_metrics.json").write_text(json.dumps(metrics))
        with (root / "cv_predictions.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        return rows

    def test_layer12_bridge_requires_both_models_and_exact_key_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); self.write_reference(root)
            summaries = {target: {"selected_alpha": 10.0, "mean_fold_mae": 2.0,
                                  "mean_demographics_fold_mae": 2.5}
                         for target in ("bmi", "height_cm")}
            predictions = [{"target": target, "speaker_id": "S", "repeat": 0, "fold": 0,
                            "prediction": 3.0, "demographics_prediction": 2.0}
                           for target in ("bmi", "height_cm")]
            config = {"targets": ["bmi", "height_cm"], "layer12_reference": {
                "model": "ssl12_demographics", "require_exact_oof_key_coverage": True,
                "mae_absolute_tolerance": 1e-9, "prediction_absolute_tolerance": 1e-8}}
            result = layers.reproduce_layer12(root, summaries, predictions, config)
            self.assertTrue(result["passed"])
            self.assertTrue(all(item["demographics_max_prediction_difference"] == 0 for item in result["checks"]))
            with self.assertRaisesRegex(layers.LayerError, "coverage"):
                layers.reproduce_layer12(root, summaries, predictions[:-1], config)

    def minimal_config(self, order):
        return {"execution_order": order, "development_splits": ["train", "val"],
                "targets": ["bmi", "height_cm"], "alpha_grid": [1e12],
                "alpha_selection": {"tie_tolerance": 1e-6},
                "paired_comparison": {"zero_tolerance": 1e-6},
                "nesting_check": {"expected_checks": len(order) * 2,
                                  "alpha": 1e12, "prediction_absolute_tolerance": 1e-6,
                                  "mae_absolute_tolerance": 1e-6},
                "layers": list(range(1, 13)),
                "layer_selection": {"near_tie_absolute_mae_band": {"bmi": .05, "height_cm": .1},
                                    "preference_layer": 6, "interpretation": "fixed"},
                "scope": "test", "cv_scores_are": "test"}

    def run_patches(self, config, bridge_passed=True, nesting_passed=True):
        data = {"ids": ["S"], "heldout_count": 1,
                "targets": {"bmi": np.asarray([1.]), "height_cm": np.asarray([2.])}}
        folds = [{"repeat": 0, "fold": 0, "train": np.asarray([0]), "validation": np.asarray([0])}]
        summary = {"selected_alpha": 1e12, "mean_fold_mae": 1.0,
                   "std_fold_mae_descriptive": 0.0, "n_folds": 1,
                   "mean_demographics_fold_mae": 1.0, "mean_delta_vs_demographics": 0.0,
                   "wins": 0, "ties": 1, "losses": 0}
        prediction = [{"target": "bmi", "speaker_id": "S", "repeat": 0, "fold": 0,
                       "prediction": 1.0, "demographics_prediction": 1.0}]
        stack = [patch.object(layers, "load_config", return_value=(config, b"x")),
                 patch.object(layers.evaluate_nisp, "load_feature_bundle",
                              return_value=([{"split": "train"}], np.zeros((1, 12, 2)), None, "fp")),
                 patch.object(layers.base, "load_development", return_value=data),
                 patch.object(layers, "validate_references", return_value=(folds, {})),
                 patch.object(layers, "protocol_lock", return_value=({"full_fold_ids": []}, "lock")),
                 patch.object(layers, "layer_matrix", return_value=np.zeros((1, 2))),
                 patch.object(layers, "prepare_layer", return_value=folds),
                 patch.object(layers, "score_layer", return_value=[]),
                 patch.object(layers, "select_alpha", return_value=(1e12, {1e12: 1.0})),
                 patch.object(layers, "summarize", return_value=(summary, [], prediction, [])),
                 patch.object(layers, "reproduce_layer12", return_value={"passed": bridge_passed, "checks": []}),
                 patch.object(layers, "nesting", return_value=[{"passed": nesting_passed}])]
        return stack

    def test_bridge_failure_stops_before_other_layers_and_metrics(self):
        config = self.minimal_config([12, 1])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            stack = self.run_patches(config, bridge_passed=False)
            with stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack[6] as prepare, stack[7], stack[8], stack[9], stack[10], stack[11]:
                with self.assertRaisesRegex(layers.LayerError, "reproduction"):
                    layers.run(Path("f"), Path("c"), Path("p"), Path("r"), output)
            self.assertEqual(prepare.call_count, 1)
            self.assertFalse((output / "layer_cv_metrics.json").exists())

    def test_nesting_failure_forbids_metric_outputs(self):
        config = self.minimal_config([12] + list(range(1, 12)))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "out"
            stack = self.run_patches(config, bridge_passed=True, nesting_passed=False)
            with stack[0], stack[1], stack[2], stack[3], stack[4], stack[5], stack[6], stack[7], stack[8], stack[9], stack[10], stack[11]:
                with self.assertRaisesRegex(layers.LayerError, "nesting"):
                    layers.run(Path("f"), Path("c"), Path("p"), Path("r"), output)
            self.assertFalse((output / "layer_cv_metrics.json").exists())

    def test_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(layers.LayerError, "refusing overwrite"):
                layers.require_new_output(Path(temporary))


if __name__ == "__main__":
    unittest.main()
