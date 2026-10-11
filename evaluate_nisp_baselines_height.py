#!/usr/bin/env python3
"""Locked development CV for constant, audio-only, combined, BMI and height models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

import audio_ridge_cv as audio_core
import cv_partial_ridge as partial_core
import evaluate_nisp
import evaluate_nisp_cv as prior_runner
import nisp_features as nf


CONFIG_SHA256 = "0f2a2710d19718548ad73360cd0249c7c72e446038701e7b737d9a53093d6448"
MODELS = ("mean", "median", "demographics", "acoustic_only", "ssl12_only",
          "acoustic_demographics", "ssl12_demographics")
AUDIO_MODELS = ("acoustic_only", "ssl12_only", "acoustic_demographics",
                "ssl12_demographics")


class BaselineError(RuntimeError):
    pass


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def _read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def require_new_output(path: Path) -> None:
    if path.exists():
        raise BaselineError("output exists; refusing overwrite")


def load_config(path: Path):
    raw = path.read_bytes()
    if _sha_bytes(raw) != CONFIG_SHA256:
        raise BaselineError("configuration does not match locked baselines/height v1")
    config = json.loads(raw)
    if config.get("models") != list(MODELS) or config.get("run_order") != ["bmi", "height_cm"]:
        raise BaselineError("configuration scope differs from locked protocol")
    return config, raw


def load_development(rows, ssl, mfcc, config):
    dev_indices = [i for i, row in enumerate(rows)
                   if row["split"] in set(config["development_splits"])]
    test_count = sum(row["split"] == "test" for row in rows)
    if len(dev_indices) != config["expected_development_speakers"]:
        raise BaselineError("development count mismatch")
    if test_count != config["expected_heldout_speakers"]:
        raise BaselineError("held-out count mismatch")
    ids, splits, age, sex, bmi, height = [], [], [], [], [], []
    for index in dev_indices:
        row = rows[index]
        ids.append(row["speaker_id"]); splits.append(row["split"])
        if row.get("sex") not in {"Female", "Male"}:
            raise BaselineError("invalid development sex")
        sex.append(0 if row["sex"] == "Female" else 1)
        try:
            age.append(float(row["age"])); bmi.append(float(row["bmi"]))
            height.append(float(row["height_cm"]))
        except (KeyError, ValueError) as exc:
            raise BaselineError("invalid development target or demographic") from exc
    values = [np.asarray(value, dtype=np.float64) for value in (age, bmi, height)]
    if len(ids) != len(set(ids)) or not all(np.isfinite(value).all() for value in values):
        raise BaselineError("development IDs or values are invalid")
    age_array, bmi_array, height_array = values
    sex_array = np.asarray(sex, dtype=np.int64)
    return {
        "ids": ids, "source_splits": splits, "sex": sex_array,
        "c": np.column_stack((np.ones(len(ids)), age_array, sex_array)),
        "targets": {"bmi": bmi_array, "height_cm": height_array},
        "audio": {"mfcc": np.asarray(mfcc[dev_indices], dtype=np.float64),
                  "ssl12": np.asarray(ssl[dev_indices, 11, :], dtype=np.float64)},
        "heldout_count": test_count,
    }


def load_prior_folds(prior_dir: Path, data, config):
    reference = config["prior_cv_reference"]
    lock_path, folds_path = prior_dir / "protocol_lock.json", prior_dir / "folds.json"
    if nf.sha256_file(lock_path) != reference["protocol_lock_sha256"]:
        raise BaselineError("prior protocol lock hash mismatch")
    if nf.sha256_file(folds_path) != reference["folds_json_sha256"]:
        raise BaselineError("prior folds hash mismatch")
    lock = json.loads(lock_path.read_text())
    folds_document = json.loads(folds_path.read_text())
    if lock.get("feature_source_fingerprint") != config["expected_feature_source_fingerprint"]:
        raise BaselineError("prior feature source fingerprint mismatch")
    if lock.get("ordered_dev_ids_hash") != reference["ordered_dev_ids_hash"]:
        raise BaselineError("prior ordered development ID hash mismatch")
    if nf.canonical_hash(data["ids"]) != reference["ordered_dev_ids_hash"]:
        raise BaselineError("current ordered development IDs differ from prior CV")
    if folds_document.get("protocol_lock_sha256") != reference["protocol_lock_sha256"]:
        raise BaselineError("fold document does not reference prior lock")
    raw_folds = folds_document.get("folds")
    if raw_folds != lock.get("full_fold_train_and_validation_ids"):
        raise BaselineError("prior lock and folds document differ")
    by_id = {speaker: index for index, speaker in enumerate(data["ids"])}
    folds, seen_by_repeat = [], {}
    for number, raw in enumerate(raw_folds):
        try:
            train = np.asarray([by_id[speaker] for speaker in raw["train_ids"]], dtype=np.int64)
            validation = np.asarray([by_id[speaker] for speaker in raw["validation_ids"]], dtype=np.int64)
        except KeyError as exc:
            raise BaselineError("prior fold contains an unknown development ID") from exc
        if (len(set(train.tolist())) != len(train) or len(set(validation.tolist())) != len(validation) or
                set(train.tolist()) & set(validation.tolist()) or
                set(train.tolist()) | set(validation.tolist()) != set(range(len(data["ids"])))):
            raise BaselineError("prior fold is not a disjoint complete development partition")
        repeat, fold = int(raw["repeat"]), int(raw["fold"])
        seen_by_repeat.setdefault(repeat, []).extend(validation.tolist())
        folds.append({"number": number, "repeat": repeat, "fold": fold,
                      "train": train, "validation": validation})
    expected_fold_count = config["cv"]["n_splits"] * config["cv"]["n_repeats"]
    if len(folds) != expected_fold_count:
        raise BaselineError("prior fold count mismatch")
    for repeat, values in seen_by_repeat.items():
        if sorted(values) != list(range(len(data["ids"]))):
            raise BaselineError(f"repeat {repeat} does not provide exactly one OOF row per speaker")
        sex_counts = [np.sum(data["sex"][fold["validation"]] == group)
                      for fold in folds if fold["repeat"] == repeat for group in (0, 1)]
        female = sex_counts[0::2]; male = sex_counts[1::2]
        if max(female) - min(female) > 1 or max(male) - min(male) > 1:
            raise BaselineError("prior folds are not reasonably sex-stratified")
    return folds, {"protocol_lock": lock, "folds_document": folds_document}


def prepare_folds(data, folds):
    prepared = []
    for fold in folds:
        train, validation = fold["train"], fold["validation"]
        c_train, c_validation = data["c"][train], data["c"][validation]
        entry = dict(fold)
        entry["c_train"] = c_train; entry["c_validation"] = c_validation
        entry["c_pinv"] = np.linalg.pinv(c_train)
        entry["pure"] = {
            "acoustic_only": audio_core.prepare_audio_fold(data["audio"]["mfcc"][train],
                                                             data["audio"]["mfcc"][validation]),
            "ssl12_only": audio_core.prepare_audio_fold(data["audio"]["ssl12"][train],
                                                          data["audio"]["ssl12"][validation]),
        }
        entry["combined"] = {
            "acoustic_demographics": partial_core.prepare_partial_fold(
                c_train, c_validation, data["audio"]["mfcc"][train], data["audio"]["mfcc"][validation]),
            "ssl12_demographics": partial_core.prepare_partial_fold(
                c_train, c_validation, data["audio"]["ssl12"][train], data["audio"]["ssl12"][validation]),
        }
        prepared.append(entry)
    return prepared


def score_target(prepared, target, alphas):
    scored = {name: [] for name in MODELS}
    for fold in prepared:
        train, validation = fold["train"], fold["validation"]
        y_train = target[train]
        predictions = {
            "mean": np.full(len(validation), y_train.mean()),
            "median": np.full(len(validation), np.median(y_train)),
            "demographics": fold["c_validation"] @ (fold["c_pinv"] @ y_train),
        }
        for name, cache in fold["pure"].items():
            predictions[name] = cache.predict_all(y_train, alphas)[0][:, :, 0]
        for name, cache in fold["combined"].items():
            predictions[name] = cache.predict_all(y_train, alphas)[0][:, :, 0]
        y_validation = target[validation]
        for name in MODELS:
            value = predictions[name]
            metrics = (partial_core.regression_metrics(y_validation, value)
                       if value.ndim == 1 else
                       [partial_core.regression_metrics(y_validation, value[:, i])
                        for i in range(len(alphas))])
            scored[name].append({"prediction": value, "metrics": metrics})
    return scored


def select_models(scored, alphas, tolerance):
    selected = {name: None for name in MODELS}
    search = {}
    for name in AUDIO_MODELS:
        scores = {float(alpha): float(np.mean([fold["metrics"][i]["mae"]
                                               for fold in scored[name]]))
                  for i, alpha in enumerate(alphas)}
        selected[name] = partial_core.choose_alpha(scores, tolerance)
        search[name] = scores
    return selected, search


def selected_outputs(target_name, data, prepared, scored, selected, search, alphas, config, lock_sha):
    fold_rows, prediction_rows, alpha_rows = [], [], []
    tolerance = config["paired_comparison"]["zero_tolerance"]
    fold_mae = {name: [] for name in MODELS}
    for name in AUDIO_MODELS:
        for alpha in alphas:
            alpha_rows.append({"lock": lock_sha, "target": target_name, "model": name,
                               "alpha": alpha, "mean_fold_mae": search[name][alpha],
                               "selected": alpha == selected[name]})
    for index, fold in enumerate(prepared):
        chosen_predictions, chosen_metrics = {}, {}
        for name in MODELS:
            if selected[name] is None:
                chosen_predictions[name] = scored[name][index]["prediction"]
                chosen_metrics[name] = scored[name][index]["metrics"]
            else:
                alpha_index = alphas.index(selected[name])
                chosen_predictions[name] = scored[name][index]["prediction"][:, alpha_index]
                chosen_metrics[name] = scored[name][index]["metrics"][alpha_index]
        demo_mae = chosen_metrics["demographics"]["mae"]
        for name in MODELS:
            metric = chosen_metrics[name]; fold_mae[name].append(metric["mae"])
            delta = metric["mae"] - demo_mae
            outcome = "reference" if name == "demographics" else (
                "win" if delta < -tolerance else "loss" if delta > tolerance else "tie")
            fold_rows.append({"lock": lock_sha, "target": target_name, "repeat": fold["repeat"],
                              "fold": fold["fold"], "model": name,
                              "selected_alpha": "" if selected[name] is None else selected[name],
                              **metric, "delta_mae_vs_demographics": delta,
                              "paired_outcome": outcome})
        for offset, dev_index in enumerate(fold["validation"]):
            actual = data["targets"][target_name][dev_index]
            row = {"lock": lock_sha, "target": target_name,
                   "speaker_id": data["ids"][dev_index], "repeat": fold["repeat"],
                   "fold": fold["fold"], "actual": actual}
            for name in MODELS:
                row[name + "_prediction"] = chosen_predictions[name][offset]
            prediction_rows.append(row)
    summaries = {}
    demo = np.asarray(fold_mae["demographics"])
    for name in MODELS:
        values = np.asarray(fold_mae[name]); delta = values - demo
        summaries[name] = {"selected_alpha": selected[name],
                           "mean_fold_mae": float(values.mean()),
                           "std_fold_mae_descriptive": float(values.std(ddof=1)),
                           "n_folds": len(values), "mean_delta_vs_demographics": float(delta.mean()),
                           "wins": int(np.sum(delta < -tolerance)),
                           "ties": int(np.sum(np.abs(delta) <= tolerance)),
                           "losses": int(np.sum(delta > tolerance))}
    return summaries, fold_rows, prediction_rows, alpha_rows


def nesting_checks(target_name, prepared, scored, alphas, config):
    settings = config["nesting_check"]; index = alphas.index(float(settings["alpha"]))
    checks = []
    for fold_index, fold in enumerate(prepared):
        references = {
            "acoustic_only": scored["mean"][fold_index], "ssl12_only": scored["mean"][fold_index],
            "acoustic_demographics": scored["demographics"][fold_index],
            "ssl12_demographics": scored["demographics"][fold_index],
        }
        for name in AUDIO_MODELS:
            prediction = scored[name][fold_index]["prediction"][:, index]
            reference = references[name]["prediction"]
            pred_diff = float(np.max(np.abs(prediction - reference)))
            mae_diff = abs(scored[name][fold_index]["metrics"][index]["mae"] -
                           references[name]["metrics"]["mae"])
            checks.append({"target": target_name, "fold_number": fold_index, "model": name,
                           "max_prediction_difference": pred_diff, "mae_difference": float(mae_diff),
                           "passed": pred_diff <= settings["prediction_absolute_tolerance"] and
                                     mae_diff <= settings["mae_absolute_tolerance"]})
    return checks


def reproduce_prior(prior_dir, summaries, predictions, config):
    settings = config["bmi_reproduction"]
    prior_metrics = json.loads((prior_dir / "cv_metrics.json").read_text())["models"]
    prior_predictions = _read_csv(prior_dir / "cv_predictions.csv")
    current = {(row["speaker_id"], str(row["repeat"]), str(row["fold"])): row for row in predictions}
    checks, passed = [], True
    for name in settings["prior_models"]:
        alpha_ok = summaries[name]["selected_alpha"] == prior_metrics[name]["selected_alpha"]
        mae_diff = abs(summaries[name]["mean_fold_mae"] - prior_metrics[name]["mean_fold_mae"])
        maximum = 0.0
        for row in prior_predictions:
            key = (row["speaker_id"], row["repeat"], row["fold"])
            if key not in current:
                raise BaselineError("prior BMI prediction key is missing")
            maximum = max(maximum, abs(float(row[name + "_prediction"]) -
                                       float(current[key][name + "_prediction"])))
        item_passed = alpha_ok and mae_diff <= settings["mae_absolute_tolerance"] and maximum <= settings["prediction_absolute_tolerance"]
        checks.append({"model": name, "alpha_matches": alpha_ok,
                       "mean_mae_difference": mae_diff,
                       "max_prediction_difference": maximum, "passed": item_passed})
        passed = passed and item_passed
    return {"passed": passed, "checks": checks}


def protocol_lock(output, config_path, raw_config, bundle_fp, data, folds, prior_dir, config):
    import sklearn, scipy
    sources = [Path(__file__), Path(audio_core.__file__), Path(partial_core.__file__),
               Path(prior_runner.__file__), Path(evaluate_nisp.__file__), Path(nf.__file__)]
    hashes = {path.name: nf.sha256_file(path.resolve()) for path in sources}
    prior_files = {name: nf.sha256_file(prior_dir / name) for name in
                   ("protocol_lock.json", "folds.json", "cv_metrics.json", "cv_predictions.csv")}
    fold_ids = [{"repeat": f["repeat"], "fold": f["fold"],
                 "train_ids": [data["ids"][i] for i in f["train"]],
                 "validation_ids": [data["ids"][i] for i in f["validation"]]} for f in folds]
    lock = {"raw_config_sha256": _sha_bytes(raw_config),
            "canonical_config_hash": nf.canonical_hash(config),
            "feature_source_fingerprint": bundle_fp,
            "ordered_dev_ids_hash": nf.canonical_hash(data["ids"]),
            "source_file_sha256": hashes, "source_code_sha256": nf.canonical_hash(hashes),
            "prior_reference_sha256": prior_files, "full_fold_ids": fold_ids,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "snapshot_event": "before_first_fit", "fit_started_at_snapshot": False}
    _write_json(output / "protocol_lock.json", lock)
    (output / "locked_config.json").write_bytes(raw_config)
    return lock, nf.sha256_file(output / "protocol_lock.json")


def run(features, config_path, prior_dir, output, plan=False):
    config, raw = load_config(config_path)
    rows, ssl, mfcc, bundle_fp = evaluate_nisp.load_feature_bundle(features)
    if bundle_fp != config["expected_feature_source_fingerprint"]:
        raise BaselineError("feature source fingerprint mismatch")
    data = load_development(rows, ssl, mfcc, config)
    folds, _prior = load_prior_folds(prior_dir, data, config)
    summary = {"development_speakers": len(data["ids"]), "heldout_count_only": data["heldout_count"],
               "folds_reused": len(folds), "targets": config["run_order"], "models": list(MODELS),
               "test_values_parsed_or_predicted": False}
    if plan:
        return summary
    require_new_output(output)
    output.mkdir(parents=True)
    lock, lock_sha = protocol_lock(output, config_path, raw, bundle_fp, data, folds, prior_dir, config)
    prepared = prepare_folds(data, folds)
    alphas = [float(value) for value in config["alpha_grid"]]
    all_summaries, all_fold, all_predictions, all_alpha, all_nesting = {}, [], [], [], []
    print("BMI reproduction started", flush=True)
    scored = score_target(prepared, data["targets"]["bmi"], alphas)
    selected, search = select_models(scored, alphas, config["selection"]["tie_tolerance"])
    summaries, fold_rows, predictions, alpha_rows = selected_outputs(
        "bmi", data, prepared, scored, selected, search, alphas, config, lock_sha)
    reproduction = reproduce_prior(prior_dir, summaries, predictions, config)
    _write_json(output / "bmi_reproduction.json", {"lock": lock_sha, **reproduction})
    if not reproduction["passed"]:
        raise BaselineError("prior BMI reproduction failed before height")
    print("BMI reproduction checked", flush=True)
    all_summaries["bmi"] = summaries; all_fold += fold_rows
    all_predictions += predictions; all_alpha += alpha_rows
    all_nesting += nesting_checks("bmi", prepared, scored, alphas, config)

    print("height started", flush=True)
    scored = score_target(prepared, data["targets"]["height_cm"], alphas)
    selected, search = select_models(scored, alphas, config["selection"]["tie_tolerance"])
    summaries, fold_rows, predictions, alpha_rows = selected_outputs(
        "height_cm", data, prepared, scored, selected, search, alphas, config, lock_sha)
    all_summaries["height_cm"] = summaries; all_fold += fold_rows
    all_predictions += predictions; all_alpha += alpha_rows
    all_nesting += nesting_checks("height_cm", prepared, scored, alphas, config)
    if len(all_nesting) != 200 or not all(item["passed"] for item in all_nesting):
        raise BaselineError("one or more large-alpha nesting checks failed")
    print("height ended", flush=True)

    metrics = {"status": "ok", "scope": "development_selection_only",
               "protocol_lock_sha256": lock_sha, "models_by_target": all_summaries,
               "optimistically_biased_by_alpha_selection": True,
               "fold_sd_is_descriptive_not_inferential": True,
               "test_metrics_or_predictions": False}
    _write_json(output / "cv_metrics.json", metrics)
    metric_rows = [{"lock": lock_sha, "target": target, "model": model, **summary}
                   for target, models in all_summaries.items() for model, summary in models.items()]
    _write_csv(output / "cv_metrics.csv",
               ("lock", "target", "model", "selected_alpha", "mean_fold_mae",
                "std_fold_mae_descriptive", "n_folds", "mean_delta_vs_demographics",
                "wins", "ties", "losses"), metric_rows)
    _write_csv(output / "cv_fold_metrics.csv",
               ("lock", "target", "repeat", "fold", "model", "selected_alpha",
                "mae", "rmse", "r2", "n", "delta_mae_vs_demographics", "paired_outcome"), all_fold)
    _write_csv(output / "cv_alpha_search.csv",
               ("lock", "target", "model", "alpha", "mean_fold_mae", "selected"), all_alpha)
    prediction_fields = ["lock", "target", "speaker_id", "repeat", "fold", "actual"] + [
        name + "_prediction" for name in MODELS]
    _write_csv(output / "cv_predictions.csv", prediction_fields, all_predictions)
    _write_json(output / "folds.json", {"lock": lock_sha, "folds": lock["full_fold_ids"]})
    _write_json(output / "nesting_checks.json", {"lock": lock_sha, "all_passed": True,
                                                  "checks": all_nesting})
    _write_json(output / "protocol.json", {"lock": lock_sha, "scope": config["scope"],
                                            "height_interpretation": config["height_interpretation"],
                                            "prior_permutation_and_sex_diagnostics_rerun": False,
                                            "final_210_person_models_fitted": False,
                                            "test_metrics_or_predictions": False})
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prior-cv", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan and args.output is None:
        parser.error("--output required unless --plan")
    try:
        result = run(args.features, args.config, args.prior_cv, args.output, args.plan)
        print(json.dumps(result, indent=2, allow_nan=False)); return 0
    except (BaselineError, audio_core.AudioRidgeError, partial_core.CVError,
            nf.FeatureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
