#!/usr/bin/env python3
"""Locked BMI development CV with unpenalized demographics and audio-only Ridge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

import cv_partial_ridge as core
import evaluate_nisp
import nisp_features as nf


CONFIG_SHA256 = "e7caf717426c638a58159319beaaa557fc8f25ebf85c40ca549bde3125813abf"
MODEL_ORDER = ("demographics", "acoustic_demographics", "ssl12_demographics")
AUDIO_MODELS = ("acoustic_demographics", "ssl12_demographics")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_locked_config(path: Path):
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != CONFIG_SHA256:
        raise core.CVError(f"configuration SHA-256 differs from locked v2: {digest}")
    config = json.loads(raw)
    if (config.get("protocol_revision") != 2 or config.get("target") != "bmi" or
            config.get("models") != list(MODEL_ORDER) or config.get("ssl_layer") != 12):
        raise core.CVError("configuration does not match the locked BMI three-model scope")
    return config, raw, digest


def development_data(rows, ssl, mfcc, config):
    """Parse development values only; held-out rows contribute only to their count."""
    dev_splits = set(config["development_splits"])
    dev_indices = [index for index, row in enumerate(rows) if row["split"] in dev_splits]
    heldout_count = sum(row["split"] == "test" for row in rows)
    if len(dev_indices) != config["expected_development_speakers"]:
        raise core.CVError("development speaker count differs from configuration")
    if heldout_count != config["expected_heldout_speakers"]:
        raise core.CVError("held-out speaker count differs from configuration")
    ids, source_splits, age, sex, target = [], [], [], [], []
    for index in dev_indices:
        row = rows[index]
        ids.append(row["speaker_id"])
        source_splits.append(row["split"])
        try:
            age.append(float(row["age"]))
            target.append(float(row["bmi"]))
        except (KeyError, ValueError) as exc:
            raise core.CVError(f"invalid development value for {row.get('speaker_id', '?')}") from exc
        if row.get("sex") not in {"Female", "Male"}:
            raise core.CVError(f"invalid development sex for {row.get('speaker_id', '?')}")
        sex.append(0 if row["sex"] == "Female" else 1)
    if len(ids) != len(set(ids)):
        raise core.CVError("duplicate development speaker IDs")
    age_array = np.asarray(age, dtype=np.float64)
    sex_array = np.asarray(sex, dtype=np.int64)
    target_array = np.asarray(target, dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (age_array, target_array)):
        raise core.CVError("development values contain NaN or Inf")
    c = np.column_stack((np.ones(len(ids)), age_array, sex_array.astype(np.float64)))
    audio = {
        "acoustic_demographics": np.asarray(mfcc[dev_indices], dtype=np.float64),
        "ssl12_demographics": np.asarray(ssl[dev_indices, 11, :], dtype=np.float64),
    }
    return {"ids": ids, "source_splits": source_splits, "age": age_array,
            "sex": sex_array, "target": target_array, "c": c, "audio": audio,
            "heldout_count": heldout_count}


def make_folds(data, config):
    from sklearn.model_selection import RepeatedStratifiedKFold

    cv = config["cv"]
    classes, counts = np.unique(data["sex"], return_counts=True)
    if len(classes) != 2:
        raise core.CVError("development CV requires exactly two sex classes")
    if int(counts.min()) < cv["n_splits"]:
        raise core.CVError("each sex class must contain at least n_splits speakers")
    splitter = RepeatedStratifiedKFold(
        n_splits=cv["n_splits"], n_repeats=cv["n_repeats"],
        random_state=cv["random_state"],
    )
    folds = []
    for number, (train, validation) in enumerate(
            splitter.split(np.zeros(len(data["ids"])), data["sex"])):
        folds.append({
            "number": number, "repeat": number // cv["n_splits"],
            "fold": number % cv["n_splits"], "train": train, "validation": validation,
        })
    return folds


def prepare_folds(data, folds):
    prepared = []
    for fold in folds:
        train, validation = fold["train"], fold["validation"]
        c_train, c_validation = data["c"][train], data["c"][validation]
        if np.linalg.matrix_rank(c_train) < 3:
            raise core.CVError("training demographic design rank is less than 3")
        entry = dict(fold)
        entry["c_train"] = c_train
        entry["c_validation"] = c_validation
        entry["c_pinv"] = np.linalg.pinv(c_train)
        entry["audio"] = {
            name: core.prepare_partial_fold(c_train, c_validation,
                                             data["audio"][name][train],
                                             data["audio"][name][validation])
            for name in AUDIO_MODELS
        }
        prepared.append(entry)
    return prepared


def score_target(prepared, target, alphas):
    """Return fold predictions; held-out y is read only after every prediction exists."""
    result = {"demographics": [], **{name: [] for name in AUDIO_MODELS}}
    for fold in prepared:
        train, validation = fold["train"], fold["validation"]
        y_train = target[train]
        a = fold["c_pinv"] @ y_train
        demographic_prediction = fold["c_validation"] @ a
        audio_predictions = {}
        for name in AUDIO_MODELS:
            prediction, _baseline = fold["audio"][name].predict_all(y_train, alphas)
            audio_predictions[name] = prediction[:, :, 0]
        # Validation targets are accessed only after all fold predictions have been formed.
        y_validation = target[validation]
        result["demographics"].append({
            "prediction": demographic_prediction,
            "metrics": core.regression_metrics(y_validation, demographic_prediction),
        })
        for name in AUDIO_MODELS:
            prediction = audio_predictions[name]
            metrics = [core.regression_metrics(y_validation, prediction[:, index])
                       for index in range(len(alphas))]
            result[name].append({"prediction": prediction, "metrics": metrics})
    return result


def selections(scored, alphas, tolerance):
    chosen = {"demographics": None}
    searches = {}
    for name in AUDIO_MODELS:
        scores = {
            float(alpha): float(np.mean([fold["metrics"][index]["mae"]
                                         for fold in scored[name]]))
            for index, alpha in enumerate(alphas)
        }
        chosen[name] = core.choose_alpha(scores, tolerance)
        searches[name] = scores
    return chosen, searches


def _source_lock(output, config_path, raw_config, config, bundle_fingerprint, data, folds):
    import sklearn
    import scipy

    source_paths = [Path(__file__).resolve(), Path(core.__file__).resolve(),
                    Path(evaluate_nisp.__file__).resolve(), Path(nf.__file__).resolve()]
    source_hashes = {path.name: nf.sha256_file(path) for path in source_paths}
    fold_ids = []
    for fold in folds:
        fold_ids.append({
            "repeat": fold["repeat"], "fold": fold["fold"],
            "train_ids": [data["ids"][index] for index in fold["train"]],
            "validation_ids": [data["ids"][index] for index in fold["validation"]],
        })
    lock = {
        "raw_config_sha256": sha256_bytes(raw_config),
        "canonical_config_hash": nf.canonical_hash(config),
        "source_code_sha256": nf.canonical_hash(source_hashes),
        "source_file_sha256": source_hashes,
        "feature_source_fingerprint": bundle_fingerprint,
        "ordered_dev_ids_hash": nf.canonical_hash(data["ids"]),
        "full_fold_train_and_validation_ids": fold_ids,
        "library_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "config_path_name": config_path.name,
        "fit_started_at_snapshot": False,
        "snapshot_event": "before_first_fit",
    }
    write_json(output / "protocol_lock.json", lock)
    (output / "locked_config.json").write_bytes(raw_config)
    return lock, nf.sha256_file(output / "protocol_lock.json")


def _fold_outputs(data, prepared, scored, chosen, searches, config, lock_sha):
    alphas = [float(value) for value in config["alpha_grid"]]
    tolerance = config["paired_comparison"]["zero_tolerance"]
    alpha_rows, fold_rows, prediction_rows = [], [], []
    for name in AUDIO_MODELS:
        for alpha in alphas:
            alpha_rows.append({"protocol_lock_sha256": lock_sha, "model": name, "alpha": alpha,
                               "mean_fold_mae": searches[name][alpha],
                               "selected": alpha == chosen[name]})
    model_fold_mae = {name: [] for name in MODEL_ORDER}
    for fold_index, fold in enumerate(prepared):
        validation = fold["validation"]
        demographic = scored["demographics"][fold_index]
        model_predictions = {"demographics": demographic["prediction"]}
        model_metrics = {"demographics": demographic["metrics"]}
        for name in AUDIO_MODELS:
            alpha_index = alphas.index(chosen[name])
            model_predictions[name] = scored[name][fold_index]["prediction"][:, alpha_index]
            model_metrics[name] = scored[name][fold_index]["metrics"][alpha_index]
        for name in MODEL_ORDER:
            metric = model_metrics[name]
            model_fold_mae[name].append(metric["mae"])
            delta = 0.0 if name == "demographics" else metric["mae"] - demographic["metrics"]["mae"]
            outcome = "reference"
            if name != "demographics":
                outcome = "win" if delta < -tolerance else "loss" if delta > tolerance else "tie"
            fold_rows.append({
                "protocol_lock_sha256": lock_sha, "repeat": fold["repeat"], "fold": fold["fold"],
                "model": name, "selected_alpha": "" if chosen[name] is None else chosen[name],
                **metric, "delta_mae_vs_demographics": delta, "paired_outcome": outcome,
            })
        for offset, dev_index in enumerate(validation):
            actual = data["target"][dev_index]
            row = {
                "protocol_lock_sha256": lock_sha, "speaker_id": data["ids"][dev_index],
                "source_split": data["source_splits"][dev_index], "repeat": fold["repeat"],
                "fold": fold["fold"], "bmi_actual": actual,
                "acoustic_demographics_alpha": chosen["acoustic_demographics"],
                "ssl12_demographics_alpha": chosen["ssl12_demographics"],
            }
            for name in MODEL_ORDER:
                predicted = model_predictions[name][offset]
                row[name + "_prediction"] = predicted
                row[name + "_error"] = predicted - actual
                row[name + "_absolute_error"] = abs(predicted - actual)
            prediction_rows.append(row)
    summaries = {}
    demo = np.asarray(model_fold_mae["demographics"])
    for name in MODEL_ORDER:
        values = np.asarray(model_fold_mae[name])
        delta = values - demo
        summaries[name] = {
            "selected_alpha": chosen[name], "mean_fold_mae": float(values.mean()),
            "std_fold_mae_descriptive": float(values.std(ddof=1)), "n_folds": len(values),
            "mean_paired_delta_vs_demographics": float(delta.mean()),
            "wins": int(np.sum(delta < -tolerance)), "ties": int(np.sum(np.abs(delta) <= tolerance)),
            "losses": int(np.sum(delta > tolerance)),
        }
    return alpha_rows, fold_rows, prediction_rows, summaries, model_fold_mae


def nesting_checks(scored, config):
    alpha = float(config["nesting_check"]["alpha"])
    alphas = [float(value) for value in config["alpha_grid"]]
    index = alphas.index(alpha)
    pred_tolerance = config["nesting_check"]["prediction_absolute_tolerance"]
    mae_tolerance = config["nesting_check"]["mae_absolute_tolerance"]
    checks = []
    for name in AUDIO_MODELS:
        for fold_index, audio in enumerate(scored[name]):
            demographic = scored["demographics"][fold_index]
            maximum = float(np.max(np.abs(audio["prediction"][:, index] - demographic["prediction"])))
            mae_difference = abs(audio["metrics"][index]["mae"] - demographic["metrics"]["mae"])
            checks.append({"model": name, "fold_number": fold_index,
                           "max_prediction_difference": maximum,
                           "mae_difference": float(mae_difference),
                           "passed": maximum <= pred_tolerance and mae_difference <= mae_tolerance})
    if config["nesting_check"]["require_all_folds_and_audio_models"] and not all(
            check["passed"] for check in checks):
        raise core.CVError("large-alpha nesting check failed")
    return checks


def permutation_diagnostic(data, prepared, observed_summaries, config):
    settings = config["permutation_diagnostic"]
    rng = np.random.default_rng(settings["random_state"])
    alphas = [float(value) for value in config["alpha_grid"]]
    tolerance = config["selection"]["tie_tolerance"]
    records = []
    for permutation in range(settings["n_permutations"]):
        y, _mapping, mapping_hash = core.stratified_permutation(data["target"], data["sex"], rng)
        scored = score_target(prepared, y, alphas)
        chosen, _searches = selections(scored, alphas, tolerance)
        demo_mae = float(np.mean([item["metrics"]["mae"] for item in scored["demographics"]]))
        models = {}
        for name in AUDIO_MODELS:
            alpha_index = alphas.index(chosen[name])
            audio_mae = float(np.mean([item["metrics"][alpha_index]["mae"] for item in scored[name]]))
            models[name] = {"selected_alpha": chosen[name],
                            "mean_fold_mae": audio_mae,
                            "delta_vs_demographics": audio_mae - demo_mae}
        records.append({"permutation": permutation, "mapping_sha256": mapping_hash, "models": models})
        print(f"permutation {permutation + 1}/{settings['n_permutations']}", flush=True)
    tails = {}
    for name in AUDIO_MODELS:
        observed = observed_summaries[name]["mean_paired_delta_vs_demographics"]
        values = [record["models"][name]["delta_vs_demographics"] for record in records]
        tails[name] = {"observed_delta": observed, "raw_permuted_deltas": values,
                       "diagnostic_lower_tail_fraction": float(np.mean(np.asarray(values) <= observed))}
    return {
        "interpretation": settings["interpretation"],
        "limitation": "Within-sex shuffling destroys the age-BMI relationship; this is not a formal p-value.",
        "records": records, "tails": tails,
    }


def sex_diagnostic(data, folds, config):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    settings = config["sex_diagnostic"]
    feature_map = {"mfcc": data["audio"]["acoustic_demographics"],
                   "ssl12": data["audio"]["ssl12_demographics"]}
    result = {}
    for name in settings["features"]:
        fold_metrics = []
        for fold in folds:
            train, validation = fold["train"], fold["validation"]
            scaler = StandardScaler().fit(feature_map[name][train])
            classifier = LogisticRegression(
                C=settings["C"], solver=settings["solver"], max_iter=settings["max_iter"],
                random_state=settings["random_state"],
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                try:
                    classifier.fit(scaler.transform(feature_map[name][train]), data["sex"][train])
                except ConvergenceWarning as exc:
                    raise core.CVError(f"sex diagnostic did not converge for {name}") from exc
            predicted = classifier.predict(scaler.transform(feature_map[name][validation]))
            counts = np.bincount(data["sex"][train], minlength=2)
            majority = np.full(len(validation), int(np.argmax(counts)))
            fold_metrics.append({
                "repeat": fold["repeat"], "fold": fold["fold"],
                "accuracy": float(accuracy_score(data["sex"][validation], predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(data["sex"][validation], predicted)),
                "majority_accuracy": float(accuracy_score(data["sex"][validation], majority)),
                "majority_balanced_accuracy": float(
                    balanced_accuracy_score(data["sex"][validation], majority)),
            })
        result[name] = {
            "folds": fold_metrics,
            "mean_accuracy": float(np.mean([item["accuracy"] for item in fold_metrics])),
            "mean_balanced_accuracy": float(np.mean([item["balanced_accuracy"] for item in fold_metrics])),
            "mean_majority_accuracy": float(np.mean([item["majority_accuracy"] for item in fold_metrics])),
            "mean_majority_balanced_accuracy": float(
                np.mean([item["majority_balanced_accuracy"] for item in fold_metrics])),
        }
    return {"demographic_inputs": False, "failure_threshold": None, "models": result}


def run(features: Path, config_path: Path, output: Path, plan_only: bool):
    config, raw_config, _digest = load_locked_config(config_path)
    rows, ssl, mfcc, bundle_fingerprint = evaluate_nisp.load_feature_bundle(features)
    data = development_data(rows, ssl, mfcc, config)
    folds = make_folds(data, config)
    summary = {"development_speakers": len(data["ids"]),
               "heldout_speakers_count_only": data["heldout_count"],
               "folds": len(folds), "models": list(MODEL_ORDER),
               "test_targets_or_predictions_used": False,
               "feature_source_fingerprint": bundle_fingerprint}
    if plan_only:
        return summary
    if output.exists():
        raise core.CVError("output already exists; refusing to overwrite")
    output.mkdir(parents=True)
    lock, lock_sha = _source_lock(output, config_path, raw_config, config,
                                  bundle_fingerprint, data, folds)
    prepared = prepare_folds(data, folds)
    alphas = [float(value) for value in config["alpha_grid"]]
    scored = score_target(prepared, data["target"], alphas)
    chosen, searches = selections(scored, alphas, config["selection"]["tie_tolerance"])
    alpha_rows, fold_rows, prediction_rows, summaries, _fold_mae = _fold_outputs(
        data, prepared, scored, chosen, searches, config, lock_sha)
    checks = nesting_checks(scored, config)
    permutation = permutation_diagnostic(data, prepared, summaries, config)
    sex = sex_diagnostic(data, folds, config)

    metrics = {
        "status": "ok", "scope": "development_selection_only",
        "result_type": "development_selection_scores",
        "score_interpretation": config["cv_scores_are"],
        "optimistically_biased_by_alpha_selection": True,
        "protocol_lock_sha256": lock_sha, "models": summaries,
        "test_metrics_or_predictions_computed": False,
    }
    write_json(output / "cv_metrics.json", metrics)
    metric_rows = [{"protocol_lock_sha256": lock_sha, "model": name, **summaries[name]}
                   for name in MODEL_ORDER]
    write_csv(output / "cv_metrics.csv",
              ("protocol_lock_sha256", "model", "selected_alpha", "mean_fold_mae",
               "std_fold_mae_descriptive", "n_folds", "mean_paired_delta_vs_demographics",
               "wins", "ties", "losses"), metric_rows)
    write_csv(output / "cv_fold_metrics.csv",
              ("protocol_lock_sha256", "repeat", "fold", "model", "selected_alpha", "mae",
               "rmse", "r2", "n", "delta_mae_vs_demographics", "paired_outcome"), fold_rows)
    write_csv(output / "cv_alpha_search.csv",
              ("protocol_lock_sha256", "model", "alpha", "mean_fold_mae", "selected"), alpha_rows)
    write_csv(output / "cv_predictions.csv",
              ("protocol_lock_sha256", "speaker_id", "source_split", "repeat", "fold", "bmi_actual",
               "demographics_prediction", "demographics_error", "demographics_absolute_error",
               "acoustic_demographics_prediction", "acoustic_demographics_error",
               "acoustic_demographics_absolute_error", "acoustic_demographics_alpha",
               "ssl12_demographics_prediction", "ssl12_demographics_error",
               "ssl12_demographics_absolute_error", "ssl12_demographics_alpha"),
              prediction_rows)
    write_json(output / "folds.json", {"protocol_lock_sha256": lock_sha,
               "folds": lock["full_fold_train_and_validation_ids"]})
    write_json(output / "nesting_checks.json", {"protocol_lock_sha256": lock_sha,
               "all_passed": all(item["passed"] for item in checks), "checks": checks})
    permutation["protocol_lock_sha256"] = lock_sha
    write_json(output / "permutation_diagnostic.json", permutation)
    sex["protocol_lock_sha256"] = lock_sha
    write_json(output / "sex_diagnostic.json", sex)
    write_json(output / "protocol.json", {
        "protocol_lock_sha256": lock_sha, "scope": config["scope"],
        "regularization": config["regularization"], "alpha_selection": config["selection"],
        "paired_comparison": config["paired_comparison"],
        "development_selection_scores_are_optimistically_biased": True,
        "fold_standard_error_used_for_inference": False,
        "test_metrics_or_predictions_computed": False,
        "final_210_speaker_model_fitted": False,
    })
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan and args.output is None:
        parser.error("--output is required unless --plan is used")
    try:
        result = run(args.features, args.config, args.output, args.plan)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (core.CVError, nf.FeatureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
