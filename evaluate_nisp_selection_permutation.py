#!/usr/bin/env python3
"""Development-only full alpha/layer-selection permutation diagnostic."""

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
import evaluate_nisp_layers as layer_runner
import nisp_features as nf


CONFIG_SHA256 = "9e8c1c3ba0242807e3bdd415e92b644b3cce64ec988aa906b7f5149aa157e887"


class PermutationError(RuntimeError):
    pass


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_config(path: Path):
    raw = path.read_bytes()
    if _sha(raw) != CONFIG_SHA256:
        raise PermutationError("configuration does not match locked permutation v1")
    config = json.loads(raw)
    if config.get("targets") != ["bmi", "height_cm"] or config.get("layers") != list(range(1, 13)):
        raise PermutationError("locked permutation scope is invalid")
    return config, raw


def build_mappings(sex: np.ndarray, config):
    settings = config["permutations"]
    count = settings["n_random_per_target"] + 1
    mappings = np.empty((2, count, len(sex)), dtype=np.int64)
    mappings[:, 0, :] = np.arange(len(sex))
    hashes = [[nf.canonical_hash(mappings[target, 0].tolist())] for target in range(2)]
    generators = [np.random.default_rng(seed) for seed in
                  np.random.SeedSequence(settings["seed"]).spawn(2)]
    for target in range(2):
        for column in range(1, count):
            mapping = np.arange(len(sex), dtype=np.int64)
            for group in np.unique(sex):
                positions = np.flatnonzero(sex == group)
                mapping[positions] = generators[target].permutation(positions)
            if not np.array_equal(sex[mapping], sex):
                raise PermutationError("permutation crossed sex strata")
            mappings[target, column] = mapping
            hashes[target].append(nf.canonical_hash(mapping.tolist()))
    return mappings, hashes


def validate_references(layer_dir: Path, baseline_dir: Path, prior_dir: Path,
                        config, data, bundle_fp):
    if bundle_fp != config["expected_feature_source_fingerprint"]:
        raise PermutationError("feature source fingerprint mismatch")
    folds, prior = base.load_prior_folds(prior_dir, data, config)
    references = config["references"]
    for name, expected in references["layer_files_sha256"].items():
        if nf.sha256_file(layer_dir / name) != expected:
            raise PermutationError(f"layer reference hash mismatch: {name}")
    for name, expected in references["baseline_files_sha256"].items():
        if nf.sha256_file(baseline_dir / name) != expected:
            raise PermutationError(f"baseline reference hash mismatch: {name}")
    return folds, prior


def target_columns(data, mappings):
    observed = np.stack([data["targets"]["bmi"], data["targets"]["height_cm"]])
    columns = np.stack([
        np.stack([observed[target][mapping] for mapping in mappings[target]], axis=1)
        for target in range(2)
    ])
    return observed, columns


def evaluate_cube(data, ssl, dev_indices, folds, mappings, config):
    targets = config["targets"]; layers = config["layers"]
    alphas = np.asarray(config["alpha_grid"], dtype=np.float64)
    observed, columns = target_columns(data, mappings)
    n_columns = columns.shape[2]
    search = np.empty((2, n_columns, 12, len(folds), len(alphas)), dtype=np.float64)
    baseline = np.empty((2, n_columns, len(folds)), dtype=np.float64)
    nesting_records = []
    huge_index = list(alphas).index(float(config["nesting_check"]["alpha"]))
    for layer in layers:
        print(f"permutation layer {layer}/12", flush=True)
        x = layer_runner.layer_matrix(ssl, dev_indices, layer)
        for fold_number, fold in enumerate(folds):
            train, validation = fold["train"], fold["validation"]
            c_train, c_validation = data["c"][train], data["c"][validation]
            cache = core.prepare_partial_fold(c_train, c_validation, x[train], x[validation])
            y_train = np.concatenate((columns[0, train, :], columns[1, train, :]), axis=1)
            y_validation = np.concatenate((columns[0, validation, :],
                                           columns[1, validation, :]), axis=1)
            predictions, demographic = cache.predict_all(y_train, alphas)
            mae = np.mean(np.abs(predictions - y_validation[:, None, :]), axis=0).T
            demographic_mae = np.mean(np.abs(demographic - y_validation), axis=0)
            for target in range(2):
                section = slice(target * n_columns, (target + 1) * n_columns)
                search[target, :, layer - 1, fold_number, :] = mae[section]
                if layer == 1:
                    baseline[target, :, fold_number] = demographic_mae[section]
                elif not np.allclose(baseline[target, :, fold_number], demographic_mae[section],
                                     rtol=0, atol=1e-12):
                    raise PermutationError("demographic baseline changed across layers")
                huge_prediction = predictions[:, huge_index, section]
                pred_diff = float(np.max(np.abs(huge_prediction - demographic[:, section])))
                mae_diff = float(np.max(np.abs(mae[section, huge_index] - demographic_mae[section])))
                nesting_records.append({"target": targets[target], "layer": layer,
                                        "fold_number": fold_number,
                                        "label_columns_checked": n_columns,
                                        "max_prediction_difference": pred_diff,
                                        "max_mae_difference": mae_diff,
                                        "passed": pred_diff <= config["nesting_check"]["prediction_absolute_tolerance"] and
                                                  mae_diff <= config["nesting_check"]["mae_absolute_tolerance"]})
    return observed, columns, search, baseline, nesting_records


def select_all(search, baseline, mapping_hashes, config):
    alphas = [float(value) for value in config["alpha_grid"]]
    rows = []
    selection_details = [[None] * search.shape[1] for _ in range(2)]
    for target_index, target in enumerate(config["targets"]):
        band = config["layer_selection"]["near_tie_absolute_mae_band"][target]
        for column in range(search.shape[1]):
            layer_scores, selected_alphas = {}, {}
            for layer in config["layers"]:
                scores = {alpha: float(search[target_index, column, layer - 1, :, index].mean())
                          for index, alpha in enumerate(alphas)}
                selected_alpha = core.choose_alpha(scores, config["alpha_selection"]["tie_tolerance"])
                selected_alphas[layer] = selected_alpha
                layer_scores[layer] = scores[selected_alpha]
            choice = layer_runner.choose_layer(layer_scores, band,
                                               config["layer_selection"]["preference_layer"])
            chosen_layer = choice["selected_layer"]
            choice["selected_alpha"] = selected_alphas[chosen_layer]
            choice["selected_alphas_by_layer"] = selected_alphas
            selection_details[target_index][column] = choice
            demo_mae = float(baseline[target_index, column].mean())
            selected_mae = float(layer_scores[chosen_layer])
            raw_layer = choice["raw_minimum_layer"]
            rows.append({"target": target, "permutation": column,
                         "kind": "observed" if column == 0 else "random",
                         "chosen_layer": chosen_layer, "chosen_alpha": selected_alphas[chosen_layer],
                         "selected_mae": selected_mae, "demographics_mae": demo_mae,
                         "gain": demo_mae - selected_mae,
                         "raw_best_gain": demo_mae - layer_scores[raw_layer],
                         "raw_best_layer": raw_layer,
                         "mapping_sha256": mapping_hashes[target_index][column]})
    return rows, selection_details


def reproduce_observed(rows, details, layer_dir: Path, config):
    metrics = json.loads((layer_dir / "layer_cv_metrics.json").read_text())["layers_by_target"]
    selections = json.loads((layer_dir / "layer_selection.json").read_text())["selection_by_target"]
    tolerance = config["observed_reproduction"]["mae_absolute_tolerance"]
    checks, passed = [], True
    for target_index, target in enumerate(config["targets"]):
        current = details[target_index][0]
        expected_selection = selections[target]
        observed_row = next(row for row in rows if row["target"] == target and
                            row["permutation"] == 0)
        selection_ok = (current["selected_layer"] == expected_selection["selected_layer"] and
                        current["selected_alpha"] == expected_selection["selected_alpha"] and
                        abs(observed_row["selected_mae"] - expected_selection["selected_score"])
                        <= tolerance)
        for layer in config["layers"]:
            expected = metrics[target][str(layer)] if str(layer) in metrics[target] else metrics[target][layer]
            alpha_ok = current["selected_alphas_by_layer"][layer] == expected["selected_alpha"]
            checks.append({"target": target, "layer": layer, "alpha_matches": alpha_ok})
            passed = passed and alpha_ok
        reference_scores = {int(layer): value["mean_fold_mae"] for layer, value in metrics[target].items()}
        score_differences = [abs(current["layer_scores"][layer] - reference_scores[layer])
                             for layer in config["layers"]]
        mae_ok = max(score_differences) <= tolerance
        reference_demo = metrics[target][str(config["layers"][0])]["mean_demographics_fold_mae"]
        demo_difference = abs(observed_row["demographics_mae"] - reference_demo)
        gain_difference = abs(observed_row["gain"] -
                              (reference_demo - expected_selection["selected_score"]))
        baseline_ok = demo_difference <= tolerance and gain_difference <= tolerance
        checks.append({"target": target, "final_selection_matches": selection_ok,
                       "max_layer_mean_mae_difference": max(score_differences),
                       "demographics_mean_mae_difference": demo_difference,
                       "gain_difference": gain_difference,
                       "mae_matches": mae_ok, "baseline_matches": baseline_ok})
        passed = passed and selection_ok and mae_ok and baseline_ok
    return {"passed": passed, "checks": checks}


def summary(rows, config):
    result = {}
    tolerance = config["statistic"]["tail_tolerance"]
    for target in config["targets"]:
        target_rows = [row for row in rows if row["target"] == target]
        observed = next(row for row in target_rows if row["permutation"] == 0)["gain"]
        random = np.asarray([row["gain"] for row in target_rows if row["permutation"] > 0])
        result[target] = {"observed_gain": observed, "random_gain_median": float(np.median(random)),
                          "random_gain_quantile_2_5": float(np.quantile(random, .025)),
                          "random_gain_quantile_97_5": float(np.quantile(random, .975)),
                          "random_gain_maximum": float(random.max()),
                          "random_gain_exceed_count": int(np.sum(random >= observed - tolerance)),
                          "random_permutations": len(random), "is_formal_p_value": False}
    return result


def protocol_lock(output, raw, config, bundle_fp, data, folds,
                  layer_dir, baseline_dir, prior_dir):
    import scipy, sklearn
    sources = [Path(__file__), Path(layer_runner.__file__), Path(base.__file__),
               Path(audio_ridge_cv.__file__), Path(core.__file__), Path(evaluate_nisp_cv.__file__),
               Path(evaluate_nisp.__file__), Path(nf.__file__)]
    root = Path(__file__).resolve().parent
    source_hashes = {(path.resolve().relative_to(root).as_posix()
                     if root in path.resolve().parents else path.resolve().as_posix()):
                     nf.sha256_file(path.resolve()) for path in sources}
    reference_hashes = {}
    for prefix, directory, names in (
        ("layer", layer_dir, config["references"]["layer_files_sha256"]),
        ("baseline", baseline_dir, config["references"]["baseline_files_sha256"]),
        ("prior", prior_dir, ("protocol_lock.json", "folds.json"))):
        for name in names:
            reference_hashes[f"{prefix}/{name}"] = nf.sha256_file(directory / name)
    fold_ids = [{"repeat": fold["repeat"], "fold": fold["fold"],
                 "train_ids": [data["ids"][i] for i in fold["train"]],
                 "validation_ids": [data["ids"][i] for i in fold["validation"]]} for fold in folds]
    lock = {"raw_config_sha256": _sha(raw), "canonical_config_hash": nf.canonical_hash(config),
            "feature_source_fingerprint": bundle_fp,
            "ordered_dev_ids_hash": nf.canonical_hash(data["ids"]),
            "source_file_sha256": source_hashes, "reference_file_sha256": reference_hashes,
            "full_fold_ids": fold_ids,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__},
            "snapshot_event": "before_first_fit", "fit_started_at_snapshot": False}
    _json(output / "protocol_lock.json", lock); (output / "locked_config.json").write_bytes(raw)
    return lock, nf.sha256_file(output / "protocol_lock.json")


def run(features, config_path, layer_dir, baseline_dir, prior_dir, output, plan=False):
    config, raw = load_config(config_path)
    rows, ssl, mfcc, bundle_fp = evaluate_nisp.load_feature_bundle(features)
    data = base.load_development(rows, ssl, mfcc, config)
    folds, _prior = validate_references(layer_dir, baseline_dir, prior_dir, config, data, bundle_fp)
    mappings, mapping_hashes = build_mappings(data["sex"], config)
    dev_indices = np.asarray([index for index, row in enumerate(rows)
                              if row["split"] in set(config["development_splits"])])
    plan_result = {"development_speakers": len(data["ids"]),
                   "heldout_count_only": data["heldout_count"], "folds_reused": len(folds),
                   "targets": config["targets"], "label_columns_per_target": mappings.shape[1],
                   "layers": config["layers"], "test_values_parsed_or_predicted": False}
    if plan:
        return plan_result
    if output.exists():
        raise PermutationError("output exists; refusing overwrite")
    output.mkdir(parents=True)
    lock, lock_sha = protocol_lock(output, raw, config, bundle_fp, data, folds,
                                   layer_dir, baseline_dir, prior_dir)
    observed, columns, search, baseline, nesting_records = evaluate_cube(
        data, ssl, dev_indices, folds, mappings, config)
    if (len(nesting_records) != config["nesting_check"]["records"] or
            not all(record["passed"] for record in nesting_records)):
        raise PermutationError("permutation nesting checks failed")
    rows_out, details = select_all(search, baseline, mapping_hashes, config)
    # Retain all layer scores for exact observed reproduction.
    for target_index in range(2):
        for column in range(search.shape[1]):
            details[target_index][column]["layer_scores"] = {
                layer: float(search[target_index, column, layer - 1, :,
                                    list(config["alpha_grid"]).index(
                                        details[target_index][column]["selected_alphas_by_layer"][layer])].mean())
                for layer in config["layers"]}
    reproduction = reproduce_observed(rows_out, details, layer_dir, config)
    if not reproduction["passed"]:
        raise PermutationError("observed selection does not reproduce layer reference")

    np.savez_compressed(output / "search_fold_mae.npz", search_fold_mae=search,
                        baseline_fold_mae=baseline, mappings=mappings,
                        observed_labels=observed, alphas=np.asarray(config["alpha_grid"]),
                        speaker_ids=np.asarray(data["ids"], dtype="U32"))
    cube_hash = nf.sha256_file(output / "search_fold_mae.npz")
    _json(output / "mapping_hashes.json", {"target_order": config["targets"],
                                            "mapping_sha256": mapping_hashes,
                                            "npz_sha256": cube_hash})
    _json(output / "nesting_checks.json", {"protocol_lock_sha256": lock_sha,
                                            "all_passed": True, "records": nesting_records,
                                            "underlying_checks": sum(
                                                record["label_columns_checked"]
                                                for record in nesting_records)})
    _json(output / "observed_reproduction.json", {"protocol_lock_sha256": lock_sha,
                                                   **reproduction})
    for row in rows_out:
        row["protocol_lock_sha256"] = lock_sha
    _csv(output / "selection_permutations.csv",
         ("protocol_lock_sha256", "target", "permutation", "kind", "chosen_layer",
          "chosen_alpha", "selected_mae", "demographics_mae", "gain",
          "raw_best_gain", "raw_best_layer", "mapping_sha256"), rows_out)
    result = {"status": "ok", "scope": config["scope"],
              "protocol_lock_sha256": lock_sha, "cube_sha256": cube_hash,
              "summary_by_target": summary(rows_out, config),
              "interpretation": config["interpretation"],
              "is_formal_p_value": False, "test_metrics_or_predictions": False}
    _json(output / "selection_permutation_summary.json", result)
    _json(output / "protocol.json", {"protocol_lock_sha256": lock_sha,
                                      "retuned_observed_settings_from_permutations": False,
                                      "test_metrics_or_predictions": False,
                                      "interpretation": config["interpretation"]})
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--layer-reference", type=Path, required=True)
    parser.add_argument("--baseline-reference", type=Path, required=True)
    parser.add_argument("--prior-cv", type=Path, required=True)
    parser.add_argument("--output", type=Path); parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan and args.output is None:
        parser.error("--output required unless --plan")
    try:
        value = run(args.features, args.config, args.layer_reference, args.baseline_reference,
                    args.prior_cv, args.output, args.plan)
        print(json.dumps(value, indent=2, allow_nan=False)); return 0
    except (PermutationError, layer_runner.LayerError, base.BaselineError,
            core.CVError, nf.FeatureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2


if __name__ == "__main__":
    raise SystemExit(main())
