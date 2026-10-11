#!/usr/bin/env python3
"""Locked 12-layer WavLM-plus-demographics development CV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import audio_ridge_cv
import cv_partial_ridge as core
import evaluate_nisp
import evaluate_nisp_baselines_height as base
import evaluate_nisp_cv
import nisp_features as nf


CONFIG_SHA256 = "ca52633b8ca7dd6f7be7e0887378a4b8fdff3637b080f8ca564764b576764654"


class LayerError(RuntimeError):
    pass


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _csv(path: Path, fields, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_config(path: Path):
    raw = path.read_bytes()
    if _sha(raw) != CONFIG_SHA256:
        raise LayerError("configuration does not match locked layer-selection v1")
    config = json.loads(raw)
    if (config.get("protocol_revision") != 2 or config.get("layers") != list(range(1, 13)) or
            config.get("execution_order") != [12] + list(range(1, 12))):
        raise LayerError("locked layer order is invalid")
    return config, raw


def layer_matrix(ssl: np.ndarray, dev_indices: np.ndarray, layer: int) -> np.ndarray:
    if layer not in range(1, 13):
        raise LayerError("layer must be in 1..12")
    return np.asarray(ssl[dev_indices, layer - 1, :], dtype=np.float64)


def choose_layer(scores, band: float, preference: int = 6):
    if (not np.isfinite(band) or band < 0 or preference not in range(1, 13)):
        raise LayerError("layer band and preference are invalid")
    if set(scores) != set(range(1, 13)) or not all(np.isfinite(value) for value in scores.values()):
        raise LayerError("layer scores must contain finite values for layers 1..12")
    minimum = min(scores.values())
    raw_minima = [layer for layer, score in scores.items() if score == minimum]
    raw_minimum = min(raw_minima)
    eligible = sorted(layer for layer, score in scores.items() if score <= minimum + band)
    selected = min(eligible, key=lambda layer: (abs(layer - preference), layer))
    return {"raw_minimum_score": float(minimum), "raw_minimum_layer": raw_minimum,
            "eligible_layers": eligible, "selected_layer": selected,
            "selected_score": float(scores[selected]), "band": band,
            "selected_score_minus_raw_minimum": float(scores[selected] - minimum),
            "preference_layer": preference}


def validate_references(prior_dir: Path, reference_dir: Path, config, data, bundle_fp):
    if bundle_fp != config["expected_feature_source_fingerprint"]:
        raise LayerError("feature source fingerprint mismatch")
    folds, prior = base.load_prior_folds(prior_dir, data, config)
    expected = config["layer12_reference"]["file_sha256"]
    actual = {name: nf.sha256_file(reference_dir / name) for name in expected}
    if actual != expected:
        raise LayerError("layer-12 reference file hash mismatch")
    return folds, {"prior": prior, "layer12_hashes": actual}


def prepare_layer(data, folds, x):
    prepared = []
    for fold in folds:
        train, validation = fold["train"], fold["validation"]
        c_train, c_validation = data["c"][train], data["c"][validation]
        entry = dict(fold)
        entry["c_train"] = c_train; entry["c_validation"] = c_validation
        entry["c_pinv"] = np.linalg.pinv(c_train)
        entry["cache"] = core.prepare_partial_fold(c_train, c_validation,
                                                    x[train], x[validation])
        prepared.append(entry)
    return prepared


def score_layer(prepared, target, alphas):
    folds = []
    for fold in prepared:
        y_train = target[fold["train"]]
        demographic = fold["c_validation"] @ (fold["c_pinv"] @ y_train)
        predictions = fold["cache"].predict_all(y_train, alphas)[0][:, :, 0]
        y_validation = target[fold["validation"]]
        folds.append({"demographics_prediction": demographic,
                      "demographics_metrics": core.regression_metrics(y_validation, demographic),
                      "predictions": predictions,
                      "metrics": [core.regression_metrics(y_validation, predictions[:, i])
                                  for i in range(len(alphas))]})
    return folds


def select_alpha(scored, alphas, tolerance):
    scores = {float(alpha): float(np.mean([fold["metrics"][index]["mae"] for fold in scored]))
              for index, alpha in enumerate(alphas)}
    return core.choose_alpha(scores, tolerance), scores


def summarize(target_name, layer, data, prepared, scored, selected_alpha, alpha_scores,
              alphas, lock_sha, tolerance):
    alpha_index = alphas.index(selected_alpha)
    fold_rows, prediction_rows = [], []
    layer_mae, demo_mae = [], []
    for number, fold in enumerate(prepared):
        metric = scored[number]["metrics"][alpha_index]
        demo = scored[number]["demographics_metrics"]
        delta = metric["mae"] - demo["mae"]
        layer_mae.append(metric["mae"]); demo_mae.append(demo["mae"])
        fold_rows.append({"lock": lock_sha, "target": target_name, "layer": layer,
                          "repeat": fold["repeat"], "fold": fold["fold"],
                          "selected_alpha": selected_alpha, **metric,
                          "demographics_mae": demo["mae"], "delta_mae_vs_demographics": delta,
                          "paired_outcome": "win" if delta < -tolerance else
                          "loss" if delta > tolerance else "tie"})
        prediction = scored[number]["predictions"][:, alpha_index]
        for offset, dev_index in enumerate(fold["validation"]):
            prediction_rows.append({"lock": lock_sha, "target": target_name, "layer": layer,
                                    "speaker_id": data["ids"][dev_index],
                                    "repeat": fold["repeat"], "fold": fold["fold"],
                                    "actual": data["targets"][target_name][dev_index],
                                    "prediction": prediction[offset],
                                    "demographics_prediction": scored[number]["demographics_prediction"][offset],
                                    "selected_alpha": selected_alpha})
    values, demo_values = np.asarray(layer_mae), np.asarray(demo_mae)
    delta = values - demo_values
    summary = {"selected_alpha": selected_alpha, "mean_fold_mae": float(values.mean()),
               "std_fold_mae_descriptive": float(values.std(ddof=1)), "n_folds": len(values),
               "mean_demographics_fold_mae": float(demo_values.mean()),
               "mean_delta_vs_demographics": float(delta.mean()),
               "wins": int(np.sum(delta < -tolerance)),
               "ties": int(np.sum(np.abs(delta) <= tolerance)),
               "losses": int(np.sum(delta > tolerance))}
    alpha_rows = [{"lock": lock_sha, "target": target_name, "layer": layer,
                   "alpha": alpha, "mean_fold_mae": alpha_scores[alpha],
                   "selected": alpha == selected_alpha} for alpha in alphas]
    return summary, fold_rows, prediction_rows, alpha_rows


def nesting(target_name, layer, scored, alphas, config):
    settings = config["nesting_check"]; index = alphas.index(float(settings["alpha"]))
    checks = []
    for number, fold in enumerate(scored):
        prediction = fold["predictions"][:, index]
        reference = fold["demographics_prediction"]
        pred_diff = float(np.max(np.abs(prediction - reference)))
        mae_diff = abs(fold["metrics"][index]["mae"] - fold["demographics_metrics"]["mae"])
        checks.append({"target": target_name, "layer": layer, "fold_number": number,
                       "max_prediction_difference": pred_diff,
                       "mae_difference": float(mae_diff),
                       "passed": pred_diff <= settings["prediction_absolute_tolerance"] and
                                 mae_diff <= settings["mae_absolute_tolerance"]})
    return checks


def reproduce_layer12(reference_dir, summaries, predictions, config):
    settings = config["layer12_reference"]
    metrics = json.loads((reference_dir / "cv_metrics.json").read_text())["models_by_target"]
    reference_rows = [row for row in _read_csv(reference_dir / "cv_predictions.csv")]
    current = {(row["target"], row["speaker_id"], str(row["repeat"]), str(row["fold"])): row
               for row in predictions}
    reference = {(row["target"], row["speaker_id"], row["repeat"], row["fold"]): row
                 for row in reference_rows}
    if len(current) != len(predictions) or len(reference) != len(reference_rows):
        raise LayerError("layer-12 bridge contains duplicate OOF keys")
    if settings["require_exact_oof_key_coverage"] and set(current) != set(reference):
        raise LayerError("layer-12 bridge OOF key coverage differs")
    checks, passed = [], True
    for target in config["targets"]:
        expected = metrics[target][settings["model"]]
        observed = summaries[target]
        alpha_ok = observed["selected_alpha"] == expected["selected_alpha"]
        mae_diff = abs(observed["mean_fold_mae"] - expected["mean_fold_mae"])
        demographic_expected = metrics[target]["demographics"]
        demographic_mae_diff = abs(observed["mean_demographics_fold_mae"] -
                                   demographic_expected["mean_fold_mae"])
        ssl_maximum = 0.0; demographic_maximum = 0.0
        target_keys = [key for key in reference if key[0] == target]
        if not target_keys:
            raise LayerError("layer-12 reference has no OOF rows for target")
        for key in target_keys:
            ssl_maximum = max(ssl_maximum,
                              abs(float(reference[key][settings["model"] + "_prediction"]) -
                                  float(current[key]["prediction"])))
            demographic_maximum = max(
                demographic_maximum,
                abs(float(reference[key]["demographics_prediction"]) -
                    float(current[key]["demographics_prediction"])))
        item = (alpha_ok and mae_diff <= settings["mae_absolute_tolerance"] and
                demographic_mae_diff <= settings["mae_absolute_tolerance"] and
                ssl_maximum <= settings["prediction_absolute_tolerance"] and
                demographic_maximum <= settings["prediction_absolute_tolerance"])
        checks.append({"target": target, "alpha_matches": alpha_ok,
                       "mean_mae_difference": mae_diff,
                       "demographics_mean_mae_difference": demographic_mae_diff,
                       "max_prediction_difference": ssl_maximum,
                       "demographics_max_prediction_difference": demographic_maximum,
                       "oof_rows_compared": len(target_keys), "passed": item})
        passed = passed and item
    return {"passed": passed, "checks": checks}


def protocol_lock(output, raw, config, bundle_fp, data, folds, prior_dir, reference_dir):
    import sklearn, scipy
    sources = [Path(__file__), Path(base.__file__), Path(audio_ridge_cv.__file__),
               Path(core.__file__), Path(evaluate_nisp_cv.__file__),
               Path(evaluate_nisp.__file__), Path(nf.__file__)]
    source_root = Path(__file__).resolve().parent
    source_hashes = {
        (path.resolve().relative_to(source_root).as_posix()
         if source_root in path.resolve().parents else path.resolve().as_posix()): nf.sha256_file(path.resolve())
        for path in sources
    }
    prior_hashes = {"prior_cv/" + name: nf.sha256_file(prior_dir / name)
                    for name in ("protocol_lock.json", "folds.json")}
    prior_hashes.update({"layer12_reference/" + name: nf.sha256_file(reference_dir / name)
                         for name in config["layer12_reference"]["file_sha256"]})
    full_folds = [{"repeat": fold["repeat"], "fold": fold["fold"],
                   "train_ids": [data["ids"][i] for i in fold["train"]],
                   "validation_ids": [data["ids"][i] for i in fold["validation"]]}
                  for fold in folds]
    lock = {"raw_config_sha256": _sha(raw), "canonical_config_hash": nf.canonical_hash(config),
            "feature_source_fingerprint": bundle_fp,
            "ordered_dev_ids_hash": nf.canonical_hash(data["ids"]),
            "source_file_sha256": source_hashes, "source_code_sha256": nf.canonical_hash(source_hashes),
            "reference_file_sha256": prior_hashes, "full_fold_ids": full_folds,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "snapshot_event": "before_first_fit", "fit_started_at_snapshot": False}
    _json(output / "protocol_lock.json", lock); (output / "locked_config.json").write_bytes(raw)
    return lock, nf.sha256_file(output / "protocol_lock.json")


def require_new_output(output: Path):
    if output.exists():
        raise LayerError("output exists; refusing overwrite")


def run(features, config_path, prior_dir, reference_dir, output, plan=False):
    config, raw = load_config(config_path)
    rows, ssl, mfcc, bundle_fp = evaluate_nisp.load_feature_bundle(features)
    data = base.load_development(rows, ssl, mfcc, config)
    folds, references = validate_references(prior_dir, reference_dir, config, data, bundle_fp)
    dev_indices = np.asarray([i for i, row in enumerate(rows)
                              if row["split"] in set(config["development_splits"])])
    summary = {"development_speakers": len(data["ids"]), "heldout_count_only": data["heldout_count"],
               "folds_reused": len(folds), "layers": config["layers"], "targets": config["targets"],
               "test_values_parsed_or_predicted": False}
    if plan:
        return summary
    require_new_output(output); output.mkdir(parents=True)
    lock, lock_sha = protocol_lock(output, raw, config, bundle_fp, data, folds, prior_dir, reference_dir)
    alphas = [float(value) for value in config["alpha_grid"]]
    tolerance = config["alpha_selection"]["tie_tolerance"]
    summaries, fold_rows, prediction_rows, alpha_rows, nest_checks = {}, [], [], [], []

    for layer in config["execution_order"]:
        print(f"layer {layer} started", flush=True)
        prepared = prepare_layer(data, folds, layer_matrix(ssl, dev_indices, layer))
        layer_summaries, layer_predictions = {}, []
        for target in config["targets"]:
            scored = score_layer(prepared, data["targets"][target], alphas)
            chosen, search = select_alpha(scored, alphas, tolerance)
            one, one_fold, one_predictions, one_alpha = summarize(
                target, layer, data, prepared, scored, chosen, search, alphas, lock_sha,
                config["paired_comparison"]["zero_tolerance"])
            layer_summaries[target] = one; layer_predictions += one_predictions
            nest_checks += nesting(target, layer, scored, alphas, config)
            if layer != 12 or target not in summaries:
                summaries.setdefault(target, {})[layer] = one
            fold_rows += one_fold; prediction_rows += one_predictions; alpha_rows += one_alpha
        if layer == 12:
            bridge = reproduce_layer12(reference_dir, layer_summaries, layer_predictions, config)
            _json(output / "layer12_reproduction.json", {"lock": lock_sha, **bridge})
            if not bridge["passed"]:
                raise LayerError("layer-12 reproduction failed before other layers")
        print(f"layer {layer} ended", flush=True)

    if len(nest_checks) != config["nesting_check"]["expected_checks"] or not all(
            item["passed"] for item in nest_checks):
        raise LayerError("one or more nesting checks failed")
    selections = {target: choose_layer(
        {layer: summaries[target][layer]["mean_fold_mae"] for layer in config["layers"]},
        config["layer_selection"]["near_tie_absolute_mae_band"][target],
        config["layer_selection"]["preference_layer"]) for target in config["targets"]}
    for target, selection in selections.items():
        chosen = selection["selected_layer"]
        selection["selected_alpha"] = summaries[target][chosen]["selected_alpha"]
    _json(output / "nesting_checks.json", {"lock": lock_sha, "all_passed": True,
                                            "checks": nest_checks})
    _json(output / "layer_selection.json", {"lock": lock_sha,
             "interpretation": config["layer_selection"]["interpretation"],
             "selection_by_target": selections})
    metrics = {"status": "ok", "scope": "development_selection_only",
               "protocol_lock_sha256": lock_sha, "layers_by_target": summaries,
               "selection_by_target": selections,
               "optimistic_from_alpha_and_layer_selection": True,
               "fold_sd_is_descriptive_not_inferential": True,
               "test_metrics_or_predictions": False}
    _json(output / "layer_cv_metrics.json", metrics)
    metric_rows = [{"lock": lock_sha, "target": target, "layer": layer, **summary}
                   for target, layers in summaries.items() for layer, summary in layers.items()]
    _csv(output / "layer_cv_metrics.csv",
         ("lock", "target", "layer", "selected_alpha", "mean_fold_mae",
          "std_fold_mae_descriptive", "n_folds", "mean_demographics_fold_mae",
          "mean_delta_vs_demographics",
          "wins", "ties", "losses"), metric_rows)
    _csv(output / "layer_cv_fold_metrics.csv",
         ("lock", "target", "layer", "repeat", "fold", "selected_alpha",
          "mae", "rmse", "r2", "n", "demographics_mae", "delta_mae_vs_demographics",
          "paired_outcome"), fold_rows)
    _csv(output / "layer_cv_alpha_search.csv",
         ("lock", "target", "layer", "alpha", "mean_fold_mae", "selected"), alpha_rows)
    _csv(output / "layer_cv_predictions.csv",
         ("lock", "target", "layer", "speaker_id", "repeat", "fold", "actual",
          "prediction", "demographics_prediction", "selected_alpha"), prediction_rows)
    _json(output / "folds.json", {"lock": lock_sha, "folds": lock["full_fold_ids"]})
    _json(output / "protocol.json", {"lock": lock_sha, "scope": config["scope"],
                                      "test_metrics_or_predictions": False,
                                      "final_210_person_models_fitted": False,
                                      "cv_scores": config["cv_scores_are"]})
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prior-cv", type=Path, required=True)
    parser.add_argument("--layer12-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path); parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan and args.output is None:
        parser.error("--output required unless --plan")
    try:
        result = run(args.features, args.config, args.prior_cv, args.layer12_reference,
                     args.output, args.plan)
        print(json.dumps(result, indent=2, allow_nan=False)); return 0
    except (LayerError, base.BaselineError, core.CVError, nf.FeatureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
