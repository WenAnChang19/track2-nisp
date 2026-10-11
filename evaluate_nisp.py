#!/usr/bin/env python3
"""First-batch BMI validation for three fixed NISP feature sets."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import nisp_features as nf


ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5)
MODEL_ORDER = ("demographics", "acoustic_demographics", "ssl12_demographics")
INPUT_WHITELISTS = {
    "demographics": ("age", "sex_Female0_Male1"),
    "acoustic_demographics": tuple([f"mfcc_{index}" for index in range(26)] +
                                     ["age", "sex_Female0_Male1"]),
    "ssl12_demographics": tuple([f"wavlm_layer12_{index}" for index in range(768)] +
                                  ["age", "sex_Female0_Male1"]),
}


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def load_feature_bundle(feature_dir: Path):
    import numpy as np

    feature_dir = feature_dir.resolve()
    try:
        metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
        config = json.loads((feature_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise nf.FeatureError(f"cannot read feature metadata: {exc}") from exc
    if metadata.get("status") != "ok":
        raise nf.FeatureError("feature extraction is incomplete")
    nf.validate_config(config)
    if config.get("limit") is not None:
        raise nf.FeatureError("pilot features cannot be evaluated; full train/val/test extraction is required")
    if metadata.get("config_fingerprint") != config.get("config_fingerprint"):
        raise nf.FeatureError("feature metadata and configuration fingerprints differ")
    inventory = config.get("selection_inventory")
    if not isinstance(inventory, list) or metadata.get("completed") != len(inventory):
        raise nf.FeatureError("feature extraction coverage is incomplete")
    utterance_shape = metadata.get("utterance_shape")
    if not isinstance(utterance_shape, list) or utterance_shape[:1] != [len(inventory)]:
        raise nf.FeatureError("utterance coverage metadata differs from selection")

    rows = nf.read_csv(feature_dir / "speakers.csv")
    try:
        with (feature_dir / "speaker_features.npz").open("rb") as handle:
            with np.load(handle, allow_pickle=False) as values:
                if set(values.files) != {
                    "ssl", "mfcc", "speaker_id", "source_fingerprint", "speaker_metadata_sha256"
                }:
                    raise nf.FeatureError("unexpected speaker feature fields")
                ssl = values["ssl"].copy()
                mfcc = values["mfcc"].copy()
                feature_speaker_ids = values["speaker_id"].astype("U32").copy()
                feature_source_fingerprint = str(values["source_fingerprint"].item())
                feature_metadata_sha256 = str(values["speaker_metadata_sha256"].item())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise nf.FeatureError(f"cannot read speaker features: {exc}") from exc
    if ssl.shape != (len(rows),) + nf.SSL_SHAPE or ssl.dtype != np.float32:
        raise nf.FeatureError("speaker SSL shape or dtype is invalid")
    if mfcc.shape != (len(rows), nf.MFCC_SIZE) or mfcc.dtype != np.float32:
        raise nf.FeatureError("speaker MFCC shape or dtype is invalid")
    if not np.isfinite(ssl).all() or not np.isfinite(mfcc).all():
        raise nf.FeatureError("speaker features contain NaN or Inf")
    if sum(int(row["num_recordings"]) for row in rows) != len(inventory):
        raise nf.FeatureError("speaker recording counts do not cover selected utterances")
    ids = [row["speaker_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise nf.FeatureError("duplicate speaker rows")
    if not np.array_equal(feature_speaker_ids, np.asarray(ids, dtype="U32")):
        raise nf.FeatureError("speaker feature order does not match speakers.csv")
    if feature_source_fingerprint != config["config_fingerprint"]:
        raise nf.FeatureError("speaker feature source fingerprint differs from configuration")
    current_metadata_sha256 = nf.sha256_file(feature_dir / "speakers.csv")
    if (feature_metadata_sha256 != current_metadata_sha256 or
            metadata.get("speaker_metadata_sha256") != current_metadata_sha256):
        raise nf.FeatureError("speaker metadata hash differs from extracted feature evidence")
    if any(row.get("split") not in {"train", "val", "test"} for row in rows):
        raise nf.FeatureError("speaker features contain an unknown split")
    split_sets = {split: {row["speaker_id"] for row in rows if row["split"] == split}
                  for split in ("train", "val", "test")}
    if any(not split_sets[split] for split in split_sets):
        raise nf.FeatureError("full evaluation input must contain train, val and test speakers")
    if any(split_sets[a] & split_sets[b]
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise nf.FeatureError("speaker appears in more than one split")
    source_fingerprint = nf.canonical_hash({
        "config_fingerprint": config["config_fingerprint"],
        "speaker_csv": nf.sha256_file(feature_dir / "speakers.csv"),
        "speaker_npz": nf.sha256_file(feature_dir / "speaker_features.npz"),
    })
    return rows, ssl, mfcc, source_fingerprint


def build_predictors(rows: Sequence[Mapping[str, str]], ssl, mfcc, indices: Sequence[int]):
    import numpy as np

    demographics: List[List[float]] = []
    for index in indices:
        row = rows[index]
        try:
            age = float(row["age"])
        except (KeyError, ValueError) as exc:
            raise nf.FeatureError(f"invalid age for {row.get('speaker_id', '?')}") from exc
        if row.get("sex") not in {"Female", "Male"} or not np.isfinite(age):
            raise nf.FeatureError(f"invalid demographics for {row.get('speaker_id', '?')}")
        demographics.append([age, 0.0 if row["sex"] == "Female" else 1.0])
    demo = np.asarray(demographics, dtype=np.float64)
    ssl12 = np.asarray(ssl[indices, 11, :], dtype=np.float64)
    acoustic = np.asarray(mfcc[indices], dtype=np.float64)
    result = {
        "demographics": demo,
        "acoustic_demographics": np.concatenate((acoustic, demo), axis=1),
        "ssl12_demographics": np.concatenate((ssl12, demo), axis=1),
    }
    for name, matrix in result.items():
        if matrix.shape[1] != len(INPUT_WHITELISTS[name]) or not np.isfinite(matrix).all():
            raise nf.FeatureError(f"invalid predictor matrix for {name}")
    return result


def _metrics(actual, predicted) -> Dict[str, float]:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "n": int(len(actual)),
    }


def select_validation_models(rows, ssl, mfcc):
    """Fit on train and select alpha on validation; test labels are never parsed."""
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_indices = [index for index, row in enumerate(rows) if row["split"] == "train"]
    val_indices = [index for index, row in enumerate(rows) if row["split"] == "val"]
    if not train_indices or not val_indices:
        raise nf.FeatureError("both train and validation speakers are required")
    try:
        train_target = np.asarray([float(rows[index]["bmi"]) for index in train_indices])
        val_target = np.asarray([float(rows[index]["bmi"]) for index in val_indices])
    except (KeyError, ValueError) as exc:
        raise nf.FeatureError("train/validation BMI labels must be numeric") from exc
    if not np.isfinite(train_target).all() or not np.isfinite(val_target).all():
        raise nf.FeatureError("train/validation BMI labels must be finite")
    train_inputs = build_predictors(rows, ssl, mfcc, train_indices)
    val_inputs = build_predictors(rows, ssl, mfcc, val_indices)
    results: Dict[str, object] = {}
    fitted: Dict[str, object] = {}
    predictions: Dict[str, object] = {}
    for name in MODEL_ORDER:
        candidates = []
        best = None
        for alpha in ALPHAS:
            pipeline = Pipeline([
                ("standardize", StandardScaler()),
                ("ridge", Ridge(alpha=alpha, solver="svd")),
            ])
            pipeline.fit(train_inputs[name], train_target)
            prediction = pipeline.predict(val_inputs[name])
            metric = _metrics(val_target, prediction)
            candidates.append({"alpha": alpha, **metric})
            key = (metric["mae"], alpha)
            if best is None or key < best[0]:
                best = (key, alpha, pipeline, prediction, metric)
        assert best is not None
        results[name] = {
            "input_columns": list(INPUT_WHITELISTS[name]),
            "chosen_alpha": best[1],
            "validation": best[4],
            "alpha_search": candidates,
        }
        fitted[name] = best[2]
        predictions[name] = best[3]
    return {
        "train_indices": train_indices, "val_indices": val_indices,
        "val_target": val_target, "results": results,
        "fitted": fitted, "predictions": predictions,
    }


def run_evaluation(feature_dir: Path, output: Path) -> Dict[str, object]:
    import joblib
    import numpy as np
    import sklearn

    rows, ssl, mfcc, source_fingerprint = load_feature_bundle(feature_dir)
    if output.exists():
        raise nf.FeatureError("evaluation output already exists; use a new directory to preserve results")
    output.mkdir(parents=True)
    selected = select_validation_models(rows, ssl, mfcc)
    protocol = {
        "schema_version": 1,
        "scope": "first_batch_validation_only",
        "target": "bmi",
        "ssl_layer": 12,
        "alpha_grid": list(ALPHAS),
        "sex_encoding": {"Female": 0, "Male": 1},
        "regularization": "StandardScaler plus ordinary Ridge on every predictor column",
        "fit_split": "train",
        "evaluation_split": "val",
        "test_outputs": False,
        "source_fingerprint": source_fingerprint,
        "input_whitelists": {key: list(value) for key, value in INPUT_WHITELISTS.items()},
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "joblib": joblib.__version__,
        },
    }
    protocol["protocol_fingerprint"] = nf.canonical_hash(protocol)
    nf.atomic_json(output / "protocol.json", protocol)

    metrics = {
        "scope": "validation_only", "target": "bmi", "ssl_layer": 12,
        "source_fingerprint": source_fingerprint,
        "protocol_fingerprint": protocol["protocol_fingerprint"],
        "models": selected["results"],
    }
    nf.atomic_json(output / "validation_metrics.json", metrics)
    metric_rows = []
    for name in MODEL_ORDER:
        result = selected["results"][name]
        metric_rows.append({"model": name, "alpha": result["chosen_alpha"],
                            **result["validation"]})
    _atomic_csv(output / "validation_metrics.csv",
                ("model", "alpha", "mae", "rmse", "r2", "n"), metric_rows)

    prediction_rows = []
    for offset, index in enumerate(selected["val_indices"]):
        row = rows[index]
        prediction_rows.append({
            "speaker_id": row["speaker_id"], "split": "val", "bmi_actual": row["bmi"],
            **{name: float(selected["predictions"][name][offset]) for name in MODEL_ORDER},
        })
    _atomic_csv(output / "validation_predictions.csv",
                ("speaker_id", "split", "bmi_actual") + MODEL_ORDER, prediction_rows)
    artifact_dir = output / "models"
    artifact_dir.mkdir()
    for name in MODEL_ORDER:
        joblib.dump(selected["fitted"][name], artifact_dir / f"{name}.joblib")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target", choices=("bmi",), default="bmi")
    parser.add_argument("--ssl-layer", type=int, choices=(12,), default=12)
    parser.add_argument("--plan", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows, _ssl, _mfcc, fingerprint = load_feature_bundle(args.features)
        summary = {
            "target": args.target, "ssl_layer": args.ssl_layer,
            "models": list(MODEL_ORDER), "train_speakers": sum(r["split"] == "train" for r in rows),
            "validation_speakers": sum(r["split"] == "val" for r in rows),
            "test_speakers_not_evaluated": sum(r["split"] == "test" for r in rows),
            "source_fingerprint": fingerprint,
        }
        if args.plan:
            print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.output is None:
            parser.error("--output is required unless --plan is used")
        result = run_evaluation(args.features, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (nf.FeatureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
