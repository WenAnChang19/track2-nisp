#!/usr/bin/env python3
"""Locked official-test prediction and later scoring for NISP body-information models."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import cv_partial_ridge as core
import nisp_features as nf
import nisp_final_inputs as final_inputs


PREDICTION_FIELDS = (
    "target", "speaker_id", "mean_prediction", "demographics_prediction",
    "mfcc_demographics_prediction", "selected_ssl_demographics_prediction",
    "ssl12_demographics_prediction",
)
MODELS = ("mean", "demographics", "mfcc_demographics",
          "selected_ssl_demographics", "ssl12_demographics")


class OfficialTestError(RuntimeError):
    pass


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_exact_csv(path: Path, fields):
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise OfficialTestError(f"unexpected columns in {path.name}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise OfficialTestError(f"cannot read {path}: {exc}") from exc


def source_hashes():
    sources = [Path(__file__), Path(final_inputs.__file__), Path(core.__file__), Path(nf.__file__)]
    root = Path(__file__).resolve().parent
    return {(path.resolve().relative_to(root).as_posix()
             if root in path.resolve().parents else path.resolve().as_posix()):
            nf.sha256_file(path.resolve()) for path in sources}


def validate_references(layer_dir: Path, baseline_dir: Path, prior_dir: Path, config):
    references = config["references"]
    hashes = {}
    for prefix, directory, expected in (
        ("layer", layer_dir, references["layer_files_sha256"]),
        ("baseline", baseline_dir, references["baseline_files_sha256"]),
    ):
        for name, digest in expected.items():
            actual = nf.sha256_file(directory / name)
            if actual != digest:
                raise OfficialTestError(f"{prefix} reference hash mismatch: {name}")
            hashes[f"{prefix}/{name}"] = actual
    prior = references["prior_cv_reference"]
    for name, key in (("protocol_lock.json", "protocol_lock_sha256"),
                      ("folds.json", "folds_json_sha256")):
        actual = nf.sha256_file(prior_dir / name)
        if actual != prior[key]:
            raise OfficialTestError(f"prior reference hash mismatch: {name}")
        hashes[f"prior/{name}"] = actual

    selected = json.loads((layer_dir / "selected_layer_settings.json").read_text())
    baseline = json.loads((baseline_dir / "cv_metrics.json").read_text())["models_by_target"]
    for target in config["targets"]:
        settings = config["settings_by_target"][target]
        reference = selected["settings_by_target"][target]
        if (reference["layer"] != settings["selected_layer"] or
                reference["alpha"] != settings["selected_ssl_alpha"] or
                baseline[target]["acoustic_demographics"]["selected_alpha"] !=
                settings["mfcc_alpha"] or
                baseline[target]["ssl12_demographics"]["selected_alpha"] !=
                settings["ssl12_alpha"]):
            raise OfficialTestError(f"locked model settings differ for {target}")
    return hashes


def build_design(data, config):
    development = set(config["development_splits"])
    dev = np.asarray([i for i, row in enumerate(data["rows"])
                      if row["split"] in development], dtype=np.int64)
    test = np.asarray([i for i, row in enumerate(data["rows"])
                       if row["split"] == "test"], dtype=np.int64)
    c = np.column_stack((np.ones(len(data["rows"])), data["age"], data["sex"]))
    if np.linalg.matrix_rank(c[dev]) < 3:
        raise OfficialTestError("development demographic design rank is less than 3")
    return dev, test, c


def fit_predictions(data, config):
    """Fit all locked models on development only and predict test rows."""
    dev, test, c = build_design(data, config)
    results = {}
    for target in config["targets"]:
        y = np.asarray(data["targets"][target], dtype=np.float64)
        if y.shape != (len(dev),) or not np.isfinite(y).all():
            raise OfficialTestError(f"invalid development target: {target}")
        settings = config["settings_by_target"][target]
        predictions = {
            "mean": np.full(len(test), y.mean()),
            "demographics": c[test] @ (np.linalg.pinv(c[dev]) @ y),
        }
        matrices = {
            "mfcc_demographics": (data["mfcc"], settings["mfcc_alpha"]),
            "selected_ssl_demographics": (
                data["ssl"][:, settings["selected_layer"] - 1, :],
                settings["selected_ssl_alpha"]),
            "ssl12_demographics": (data["ssl"][:, 11, :], settings["ssl12_alpha"]),
        }
        for model, (matrix, alpha) in matrices.items():
            cache = core.prepare_partial_fold(c[dev], c[test], matrix[dev], matrix[test])
            predictions[model] = cache.predict_all(y, [float(alpha)])[0][:, 0, 0]
        if set(predictions) != set(MODELS) or any(
                value.shape != (len(test),) or not np.isfinite(value).all()
                for value in predictions.values()):
            raise OfficialTestError(f"invalid fitted predictions for {target}")
        results[target] = predictions
    return test, results


def prediction_rows_and_audit(data, test_indices, predictions, config):
    rows, ranges = [], {}
    test_ids = [data["ids"][i] for i in test_indices]
    for target in config["targets"]:
        target_rows = []
        for offset, speaker_id in enumerate(test_ids):
            row = {"target": target, "speaker_id": speaker_id}
            for model in MODELS:
                row[model + "_prediction"] = float(predictions[target][model][offset])
            rows.append(row); target_rows.append(row)
        dev_values = data["targets"][target]
        target_ranges = {}
        for model in MODELS:
            values = np.asarray([row[model + "_prediction"] for row in target_rows])
            target_ranges[model] = {
                "prediction_min": float(values.min()), "prediction_max": float(values.max()),
                "development_target_min": float(dev_values.min()),
                "development_target_max": float(dev_values.max()),
                "below_development_range": int(np.sum(values < dev_values.min())),
                "above_development_range": int(np.sum(values > dev_values.max())),
                "clipped_or_excluded": False,
            }
        ranges[target] = target_ranges
    expected_keys = [(target, speaker) for target in config["targets"] for speaker in test_ids]
    actual_keys = [(row["target"], row["speaker_id"]) for row in rows]
    passed = (actual_keys == expected_keys and len(actual_keys) == len(set(actual_keys)) and
              all(np.isfinite(float(row[field])) for row in rows
                  for field in PREDICTION_FIELDS[2:]))
    return rows, {"status": "passed" if passed else "failed",
                  "prediction_columns": list(PREDICTION_FIELDS),
                  "target_order": config["targets"], "test_ids": test_ids,
                  "test_ids_hash": nf.canonical_hash(test_ids),
                  "rows": len(rows), "finite": True if passed else False,
                  "exact_target_major_test_id_order": passed,
                  "range_audit_only_no_clipping": ranges}


def prediction_lock(output, input_dir, raw, config, reference_hashes,
                    layer_dir, baseline_dir, prior_dir):
    import scipy
    import sklearn
    input_names = ("view_manifest.json", "predictors.csv", "development_targets.csv",
                   "validated_features.npz", "locked_config.json")
    lock = {
        "snapshot_event": "before_first_fit", "fit_started_at_snapshot": False,
        "raw_config_sha256": final_inputs.sha256_bytes(raw),
        "canonical_config_hash": nf.canonical_hash(config),
        "source_file_sha256": source_hashes(),
        "input_directory": str(input_dir.resolve()),
        "input_file_sha256": {name: nf.sha256_file(input_dir / name) for name in input_names},
        "reference_file_sha256": reference_hashes,
        "reference_directories": {"layer": str(layer_dir.resolve()),
                                  "baseline": str(baseline_dir.resolve()),
                                  "prior": str(prior_dir.resolve())},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "test_targets_loaded_at_snapshot": False,
    }
    write_json(output / "prediction_lock.json", lock)
    (output / "locked_config.json").write_bytes(raw)
    return lock, nf.sha256_file(output / "prediction_lock.json")


def run_predict(input_dir: Path, config_path: Path, layer_dir: Path, baseline_dir: Path,
                prior_dir: Path, output: Optional[Path], plan=False):
    config, raw = final_inputs.load_config(config_path)
    data = final_inputs.load_prepared_inputs(input_dir, config)
    reference_hashes = validate_references(layer_dir, baseline_dir, prior_dir, config)
    plan_result = {"development_speakers": len(data["dev_ids"]),
                   "test_speakers": len(data["test_ids"]), "targets": config["targets"],
                   "models": config["model_order"], "test_targets_loaded": False}
    if plan:
        return plan_result
    if output is None:
        raise OfficialTestError("output is required outside plan mode")
    if output.exists():
        raise OfficialTestError("prediction output exists; refusing overwrite")
    output.mkdir(parents=True)
    lock, lock_sha = prediction_lock(output, input_dir, raw, config, reference_hashes,
                                     layer_dir, baseline_dir, prior_dir)
    test_indices, predictions = fit_predictions(data, config)
    rows, audit = prediction_rows_and_audit(data, test_indices, predictions, config)
    if audit["status"] != "passed":
        raise OfficialTestError("prediction quality audit failed")
    audit["prediction_lock_sha256"] = lock_sha
    write_csv(output / "test_predictions.csv", PREDICTION_FIELDS, rows)
    write_json(output / "prediction_audit.json", audit)
    manifest = {
        "status": "ready_for_scoring", "prediction_lock_sha256": lock_sha,
        "locked_config_sha256": nf.sha256_file(output / "locked_config.json"),
        "prediction_csv_sha256": nf.sha256_file(output / "test_predictions.csv"),
        "prediction_audit_sha256": nf.sha256_file(output / "prediction_audit.json"),
        "source_file_sha256": lock["source_file_sha256"],
        "test_targets_loaded": False,
    }
    write_json(output / "prediction_manifest.json", manifest)
    return {"status": "ready_for_scoring", "prediction_rows": len(rows),
            "prediction_csv_sha256": manifest["prediction_csv_sha256"],
            "test_targets_loaded": False}


def validate_prediction_artifacts(prediction_dir: Path, config_path: Path):
    config, raw = final_inputs.load_config(config_path)
    try:
        manifest = json.loads((prediction_dir / "prediction_manifest.json").read_text())
        lock = json.loads((prediction_dir / "prediction_lock.json").read_text())
        audit = json.loads((prediction_dir / "prediction_audit.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialTestError(f"cannot read prediction evidence: {exc}") from exc
    if (manifest.get("status") != "ready_for_scoring" or
            manifest.get("test_targets_loaded") is not False or
            manifest.get("prediction_lock_sha256") !=
            nf.sha256_file(prediction_dir / "prediction_lock.json") or
            manifest.get("locked_config_sha256") != final_inputs.CONFIG_SHA256 or
            nf.sha256_file(prediction_dir / "locked_config.json") != final_inputs.CONFIG_SHA256 or
            lock.get("raw_config_sha256") != final_inputs.sha256_bytes(raw)):
        raise OfficialTestError("prediction manifest or configuration binding differs")
    if (manifest.get("prediction_csv_sha256") !=
            nf.sha256_file(prediction_dir / "test_predictions.csv") or
            manifest.get("prediction_audit_sha256") !=
            nf.sha256_file(prediction_dir / "prediction_audit.json")):
        raise OfficialTestError("prediction or audit hash mismatch")
    if audit.get("status") != "passed" or audit.get("prediction_lock_sha256") != \
            manifest["prediction_lock_sha256"]:
        raise OfficialTestError("prediction quality audit did not pass")
    if lock.get("source_file_sha256") != source_hashes():
        raise OfficialTestError("computing source changed after prediction")
    if manifest.get("source_file_sha256") != lock.get("source_file_sha256"):
        raise OfficialTestError("prediction manifest source snapshot differs")
    input_dir = Path(lock["input_directory"])
    for name, expected in lock.get("input_file_sha256", {}).items():
        if nf.sha256_file(input_dir / name) != expected:
            raise OfficialTestError(f"prepared input changed after prediction: {name}")
    reference_dirs = lock.get("reference_directories", {})
    for key, expected in lock.get("reference_file_sha256", {}).items():
        prefix, name = key.split("/", 1)
        if prefix not in reference_dirs or nf.sha256_file(Path(reference_dirs[prefix]) / name) != expected:
            raise OfficialTestError(f"development reference changed after prediction: {key}")
    rows = read_exact_csv(prediction_dir / "test_predictions.csv", PREDICTION_FIELDS)
    data = final_inputs.load_prepared_inputs(input_dir, config)
    expected_keys = [(target, speaker) for target in config["targets"]
                     for speaker in data["test_ids"]]
    if ([(row["target"], row["speaker_id"]) for row in rows] != expected_keys or
            len(rows) != len(set(expected_keys)) or
            audit.get("prediction_columns") != list(PREDICTION_FIELDS) or
            audit.get("target_order") != config["targets"] or
            audit.get("test_ids") != data["test_ids"]):
        raise OfficialTestError("prediction target/ID coverage or order differs")
    for row in rows:
        try:
            values = [float(row[model + "_prediction"]) for model in MODELS]
        except ValueError as exc:
            raise OfficialTestError("prediction value is not numeric") from exc
        if not np.isfinite(values).all():
            raise OfficialTestError("prediction value is not finite")
    return config, raw, manifest, lock, audit, rows, data


def load_test_targets(path: Path, expected_sha: str, expected_ids, targets):
    if nf.sha256_file(path) != expected_sha:
        raise OfficialTestError("official test target source hash mismatch")
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(("speaker_id", "split", *targets)).issubset(reader.fieldnames or []):
                raise OfficialTestError("official target source lacks required columns")
            rows = []
            for row in reader:
                if row["split"] != "test":
                    continue
                values = {}
                for target in targets:
                    try:
                        value = float(row[target])
                    except ValueError as exc:
                        raise OfficialTestError("official test target is invalid") from exc
                    if not np.isfinite(value):
                        raise OfficialTestError("official test target is non-finite")
                    values[target] = value
                rows.append((row["speaker_id"], values))
    except OSError as exc:
        raise OfficialTestError(f"cannot read official targets: {exc}") from exc
    if [row[0] for row in rows] != list(expected_ids):
        raise OfficialTestError("official test target IDs or order differ")
    return {target: np.asarray([row[1][target] for row in rows]) for target in targets}


def predictions_by_target(rows, config):
    result = {}
    for target in config["targets"]:
        selected = [row for row in rows if row["target"] == target]
        result[target] = {model: np.asarray([float(row[model + "_prediction"])
                                             for row in selected]) for model in MODELS}
    return result


def paired_bootstrap(actual, predictions, indices, candidate, reference):
    actual = np.asarray(actual, dtype=np.float64)
    candidate_error = np.abs(actual - np.asarray(predictions[candidate], dtype=np.float64))
    reference_error = np.abs(actual - np.asarray(predictions[reference], dtype=np.float64))
    if (indices.ndim != 2 or indices.shape[1] != len(actual) or
            np.any(indices < 0) or np.any(indices >= len(actual))):
        raise OfficialTestError("bootstrap speaker indices are invalid")
    differences = candidate_error - reference_error
    return differences[indices].mean(axis=1), float(differences.mean())


def bootstrap_outputs(actuals, predictions, config):
    settings = config["bootstrap"]
    n = len(next(iter(actuals.values())))
    indices = np.random.default_rng(settings["seed"]).integers(
        0, n, size=(settings["n_resamples"], n), dtype=np.int64)
    records, distributions = [], []
    comparisons = [(item["target"], item["candidate"], item["reference"], "primary")
                   for item in config["primary_comparisons"]]
    comparisons += [(target, item["candidate"], item["reference"], "secondary")
                    for target in config["targets"]
                    for item in config["secondary_comparisons"]]
    for target, candidate, reference, kind in comparisons:
        distribution, point = paired_bootstrap(
            actuals[target], predictions[target], indices, candidate, reference)
        lower, upper = np.quantile(distribution,
                                   settings["individual_percentile_quantiles"])
        family_lower = family_upper = None
        if kind == "primary":
            family_lower, family_upper = np.quantile(
                distribution, settings["two_target_bonferroni_two_sided_percentile_quantiles"])
        records.append({
            "target": target, "kind": kind, "candidate": candidate, "reference": reference,
            "delta_mae_candidate_minus_reference": point,
            "ci95_lower": float(lower), "ci95_upper": float(upper),
            "individual_support": bool(upper < 0) if kind == "primary" else None,
            "family_ci97_5_lower": None if family_lower is None else float(family_lower),
            "family_ci97_5_upper": None if family_upper is None else float(family_upper),
            "family_adjusted_support": bool(family_upper < 0) if kind == "primary" else None,
            "interpretation": "confirmatory_primary" if kind == "primary" else
                              "descriptive_secondary",
        })
        distributions.append(distribution)
    return indices, records, np.asarray(distributions)


def metric_outputs(actuals, predictions, config):
    result, rows = {}, []
    for target in config["targets"]:
        result[target] = {}
        for model in config["model_order"]:
            metric = core.regression_metrics(actuals[target], predictions[target][model])
            result[target][model] = metric
            rows.append({"target": target, "model": model, **metric})
    return result, rows


def group_outputs(actuals, predictions, data, config):
    test_rows = [row for row in data["rows"] if row["split"] == "test"]
    result = []; csv_rows = []
    for field in config["group_analysis"]["fields"]:
        for category in config["group_analysis"]["categories"][field]:
            mask = np.asarray([row[field] == category for row in test_rows])
            n = int(mask.sum())
            for target in config["targets"]:
                demo_error = (float(np.mean(np.abs(actuals[target][mask] -
                                                   predictions[target]["demographics"][mask])))
                              if n else None)
                for model in config["model_order"]:
                    if n:
                        metric = core.regression_metrics(actuals[target][mask],
                                                         predictions[target][model][mask])
                        delta = metric["mae"] - demo_error
                        mae, rmse = metric["mae"], metric["rmse"]
                    else:
                        mae = rmse = delta = None
                    item = {"group_field": field, "group_value": category,
                            "target": target, "model": model, "n": n,
                            "mae": mae, "rmse": rmse,
                            "delta_mae_vs_demographics": delta}
                    result.append(item)
                    csv_rows.append({key: "" if value is None else value
                                     for key, value in item.items()})
    return result, csv_rows


def score_lock(output, raw, prediction_dir, prediction_manifest, target_path,
               target_sha, data):
    lock = {
        "snapshot_event": "after_locked_predictions_before_test_target_parsing",
        "raw_config_sha256": final_inputs.sha256_bytes(raw),
        "prediction_directory": str(prediction_dir.resolve()),
        "prediction_manifest_sha256": nf.sha256_file(prediction_dir / "prediction_manifest.json"),
        "prediction_csv_sha256": prediction_manifest["prediction_csv_sha256"],
        "official_target_source": str(target_path.resolve()),
        "official_target_source_sha256": target_sha,
        "test_ids_hash": nf.canonical_hash(data["test_ids"]),
        "source_file_sha256": source_hashes(),
        "test_targets_loaded_at_snapshot": False,
        "model_fit_or_selection_during_score": False,
    }
    write_json(output / "score_lock.json", lock)
    (output / "locked_config.json").write_bytes(raw)
    return lock, nf.sha256_file(output / "score_lock.json")


def run_score(prediction_dir: Path, config_path: Path, test_target_source: Path,
              output: Path):
    if output.exists():
        raise OfficialTestError("score output exists; refusing overwrite")
    # Every immutable prediction and source gate is checked before target-source hashing or parsing.
    config, raw, manifest, _prediction_lock, _audit, rows, data = \
        validate_prediction_artifacts(prediction_dir, config_path)
    expected_source_sha = data["manifest"]["source_evidence_sha256"]["speakers.csv"]
    if nf.sha256_file(test_target_source) != expected_source_sha:
        raise OfficialTestError("official test target source hash mismatch")
    output.mkdir(parents=True)
    _lock, lock_sha = score_lock(output, raw, prediction_dir, manifest,
                                  test_target_source, expected_source_sha, data)
    actuals = load_test_targets(test_target_source, expected_source_sha,
                                data["test_ids"], config["targets"])
    predictions = predictions_by_target(rows, config)
    metrics, metric_rows = metric_outputs(actuals, predictions, config)
    indices, bootstrap_rows, bootstrap_distributions = bootstrap_outputs(
        actuals, predictions, config)
    groups, group_rows = group_outputs(actuals, predictions, data, config)
    scored_rows = []
    for target in config["targets"]:
        for index, speaker_id in enumerate(data["test_ids"]):
            row = {"target": target, "speaker_id": speaker_id,
                   "actual": float(actuals[target][index])}
            for model in MODELS:
                row[model + "_prediction"] = float(predictions[target][model][index])
            scored_rows.append(row)
    write_json(output / "test_metrics.json", {
        "status": "ok", "score_lock_sha256": lock_sha,
        "metrics_by_target_and_model": metrics,
        "test_speakers": len(data["test_ids"]), "targets": config["targets"],
        "models": config["model_order"],
    })
    write_csv(output / "test_metrics.csv", ("target", "model", "mae", "rmse", "r2", "n"),
              metric_rows)
    write_json(output / "bootstrap_comparisons.json", {
        "score_lock_sha256": lock_sha, "method": config["bootstrap"]["method"],
        "same_draws_for_all_models_and_targets": True,
        "comparisons": bootstrap_rows,
        "uncertainty_scope": config["bootstrap"]["uncertainty_scope"],
    })
    bootstrap_fields = ("target", "kind", "candidate", "reference",
                        "delta_mae_candidate_minus_reference", "ci95_lower", "ci95_upper",
                        "individual_support", "family_ci97_5_lower", "family_ci97_5_upper",
                        "family_adjusted_support", "interpretation")
    write_csv(output / "bootstrap_comparisons.csv", bootstrap_fields,
              [{key: "" if value is None else value for key, value in row.items()}
               for row in bootstrap_rows])
    np.savez_compressed(output / "bootstrap_resamples.npz", speaker_indices=indices,
                        speaker_ids=np.asarray(data["test_ids"], dtype="U32"),
                        comparison_distributions=bootstrap_distributions)
    write_json(output / "bootstrap_evidence.json", {
        "resample_npz_sha256": nf.sha256_file(output / "bootstrap_resamples.npz"),
        "speaker_index_matrix_shape": list(indices.shape),
        "speaker_index_matrix_sha256": nf.canonical_hash(indices.tolist()),
        "comparison_order": [{key: row[key] for key in
                              ("target", "kind", "candidate", "reference")}
                             for row in bootstrap_rows],
    })
    write_json(output / "group_metrics.json", {
        "score_lock_sha256": lock_sha, "rows": groups,
        "small_group_note": config["group_analysis"]["small_group_note"],
        "no_group_model_selection": True,
    })
    group_fields = ("group_field", "group_value", "target", "model", "n",
                    "mae", "rmse", "delta_mae_vs_demographics")
    write_csv(output / "group_metrics.csv", group_fields, group_rows)
    write_csv(output / "test_scored_predictions.csv",
              ("target", "speaker_id", "actual") + PREDICTION_FIELDS[2:], scored_rows)
    write_json(output / "protocol.json", {
        "score_lock_sha256": lock_sha, "test_driven_model_selection": False,
        "all_locked_targets_models_and_groups_reported": True,
        "scientific_wording": config["scientific_wording"],
    })
    return {"status": "ok", "test_speakers": len(data["test_ids"]),
            "metric_rows": len(metric_rows), "bootstrap_comparisons": len(bootstrap_rows),
            "group_rows": len(groups)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    predict = subparsers.add_parser("predict")
    predict.add_argument("--inputs", type=Path, required=True)
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--layer-reference", type=Path, required=True)
    predict.add_argument("--baseline-reference", type=Path, required=True)
    predict.add_argument("--prior-cv", type=Path, required=True)
    predict.add_argument("--output", type=Path)
    predict.add_argument("--plan", action="store_true")
    score = subparsers.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--config", type=Path, required=True)
    score.add_argument("--test-targets", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "predict" and not args.plan and args.output is None:
        raise SystemExit("--output is required outside --plan")
    try:
        if args.command == "predict":
            result = run_predict(args.inputs, args.config, args.layer_reference,
                                 args.baseline_reference, args.prior_cv, args.output, args.plan)
        else:
            result = run_score(args.predictions, args.config, args.test_targets, args.output)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (OfficialTestError, final_inputs.FinalInputError, core.CVError,
            nf.FeatureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
