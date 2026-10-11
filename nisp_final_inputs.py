#!/usr/bin/env python3
"""Prepare and validate physically label-separated inputs for official testing."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

import nisp_features as nf


CONFIG_SHA256 = "52e9ff6927d9d0da57e4c29ffd821a11eab9472c38c551214a1094cb506ced21"
PREDICTOR_FIELDS = ("speaker_id", "split", "age", "sex", "language")
DEVELOPMENT_TARGET_FIELDS = ("speaker_id", "split", "bmi", "height_cm")
NPZ_FIELDS = {"ssl", "mfcc", "speaker_id", "source_fingerprint", "speaker_metadata_sha256"}


class FinalInputError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_exact_csv(path: Path, fields):
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != tuple(fields):
                raise FinalInputError(f"unexpected columns in {path.name}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise FinalInputError(f"cannot read {path}: {exc}") from exc


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path):
    raw = path.read_bytes()
    if sha256_bytes(raw) != CONFIG_SHA256:
        raise FinalInputError("configuration does not match locked official-test v2")
    config = json.loads(raw)
    if (config.get("protocol_revision") != 2 or
            config.get("targets") != ["bmi", "height_cm"] or
            config.get("model_order") != ["mean", "demographics", "mfcc_demographics",
                                           "selected_ssl_demographics",
                                           "ssl12_demographics"]):
        raise FinalInputError("locked official-test scope is invalid")
    return config, raw


def project_rows(rows, config):
    """Project allowed predictors for all rows and targets only for development rows."""
    development = set(config["development_splits"])
    predictors, targets, ids = [], [], []
    counts = {"development": 0, "test": 0}
    allowed_languages = set(config["group_analysis"]["categories"]["language"])
    for row in rows:
        try:
            speaker_id = row["speaker_id"]
            split = row["split"]
            age = float(row["age"])
            sex = row["sex"]
            language = row["language"]
        except (KeyError, TypeError, ValueError) as exc:
            raise FinalInputError("invalid predictor row") from exc
        if (not speaker_id or split not in development | {"test"} or
                not np.isfinite(age) or sex not in {"Female", "Male"} or
                language not in allowed_languages):
            raise FinalInputError(f"invalid predictor values for {speaker_id or '?'}")
        if speaker_id in ids:
            raise FinalInputError(f"duplicate speaker ID: {speaker_id}")
        ids.append(speaker_id)
        predictors.append({"speaker_id": speaker_id, "split": split, "age": row["age"],
                           "sex": sex, "language": language})
        if split in development:
            try:
                bmi, height = float(row["bmi"]), float(row["height_cm"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FinalInputError(f"invalid development targets for {speaker_id}") from exc
            if not np.isfinite([bmi, height]).all():
                raise FinalInputError(f"non-finite development targets for {speaker_id}")
            targets.append({"speaker_id": speaker_id, "split": split,
                            "bmi": row["bmi"], "height_cm": row["height_cm"]})
            counts["development"] += 1
        else:
            counts["test"] += 1
    if (counts["development"] != config["expected_development_speakers"] or
            counts["test"] != config["expected_test_speakers"]):
        raise FinalInputError("development or test speaker count mismatch")
    return predictors, targets


def validate_source_bundle(feature_dir: Path, config):
    feature_dir = feature_dir.resolve()
    try:
        metadata = json.loads((feature_dir / "metadata.json").read_text(encoding="utf-8"))
        feature_config = json.loads((feature_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalInputError(f"cannot read source feature evidence: {exc}") from exc
    if metadata.get("status") != "ok" or feature_config.get("limit") is not None:
        raise FinalInputError("source feature extraction is incomplete or is a pilot")
    nf.validate_config(feature_config)
    if metadata.get("config_fingerprint") != feature_config.get("config_fingerprint"):
        raise FinalInputError("source feature configuration fingerprints differ")
    speaker_csv = feature_dir / "speakers.csv"
    speaker_npz = feature_dir / "speaker_features.npz"
    evidence = {
        "metadata.json": nf.sha256_file(feature_dir / "metadata.json"),
        "config.json": nf.sha256_file(feature_dir / "config.json"),
        "speakers.csv": nf.sha256_file(speaker_csv),
        "speaker_features.npz": nf.sha256_file(speaker_npz),
    }
    source_fingerprint = nf.canonical_hash({
        "config_fingerprint": feature_config["config_fingerprint"],
        "speaker_csv": evidence["speakers.csv"],
        "speaker_npz": evidence["speaker_features.npz"],
    })
    if source_fingerprint != config["expected_feature_source_fingerprint"]:
        raise FinalInputError("source feature fingerprint mismatch")
    rows = nf.read_csv(speaker_csv)
    try:
        with np.load(speaker_npz, allow_pickle=False) as values:
            if set(values.files) != NPZ_FIELDS:
                raise FinalInputError("unexpected source speaker feature fields")
            ssl = values["ssl"]
            mfcc = values["mfcc"]
            ids = values["speaker_id"].astype("U32")
            embedded_source = str(values["source_fingerprint"].item())
            embedded_metadata = str(values["speaker_metadata_sha256"].item())
            if (ssl.shape != (len(rows),) + nf.SSL_SHAPE or ssl.dtype != np.float32 or
                    mfcc.shape != (len(rows), nf.MFCC_SIZE) or mfcc.dtype != np.float32 or
                    not np.isfinite(ssl).all() or not np.isfinite(mfcc).all()):
                raise FinalInputError("source speaker feature arrays are invalid")
            row_ids = np.asarray([row.get("speaker_id", "") for row in rows], dtype="U32")
            if not np.array_equal(ids, row_ids):
                raise FinalInputError("source feature and speaker row order differ")
            if (embedded_source != feature_config["config_fingerprint"] or
                    embedded_metadata != evidence["speakers.csv"] or
                    metadata.get("speaker_metadata_sha256") != evidence["speakers.csv"]):
                raise FinalInputError("source NPZ binding evidence differs")
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise FinalInputError(f"cannot validate source speaker features: {exc}") from exc
    return rows, speaker_npz, source_fingerprint, evidence, feature_config["config_fingerprint"]


def atomic_copy(source: Path, destination: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=str(destination.parent))
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def prepare(feature_dir: Path, config_path: Path, output: Optional[Path], plan=False):
    config, raw = load_config(config_path)
    rows, npz_path, source_fp, evidence, extraction_fp = validate_source_bundle(
        feature_dir, config)
    predictors, development_targets = project_rows(rows, config)
    test_ids = [row["speaker_id"] for row in predictors if row["split"] == "test"]
    dev_ids = [row["speaker_id"] for row in predictors if row["split"] in
               set(config["development_splits"])]
    result = {"development_speakers": len(dev_ids), "test_speakers": len(test_ids),
              "feature_source_fingerprint": source_fp,
              "predictor_columns": list(PREDICTOR_FIELDS),
              "development_target_columns": list(DEVELOPMENT_TARGET_FIELDS),
              "test_target_fields_accessed": False}
    if plan:
        return result
    if output is None:
        raise FinalInputError("output is required outside plan mode")
    if output.exists():
        raise FinalInputError("output exists; refusing overwrite")
    output.mkdir(parents=True)
    write_csv(output / "predictors.csv", PREDICTOR_FIELDS, predictors)
    write_csv(output / "development_targets.csv", DEVELOPMENT_TARGET_FIELDS,
              development_targets)
    atomic_copy(npz_path, output / "validated_features.npz")
    (output / "locked_config.json").write_bytes(raw)
    prepared_hashes = {name: nf.sha256_file(output / name) for name in
                       ("predictors.csv", "development_targets.csv",
                        "validated_features.npz", "locked_config.json")}
    manifest = {
        "status": "ok", "scope": "physically_label_separated_official_test_inputs",
        "raw_config_sha256": sha256_bytes(raw),
        "feature_source_fingerprint": source_fp,
        "feature_extraction_config_fingerprint": extraction_fp,
        "source_evidence_sha256": evidence,
        "prepared_file_sha256": prepared_hashes,
        "predictor_columns": list(PREDICTOR_FIELDS),
        "development_target_columns": list(DEVELOPMENT_TARGET_FIELDS),
        "speaker_order_hash": nf.canonical_hash([row["speaker_id"] for row in predictors]),
        "development_ids_hash": nf.canonical_hash(dev_ids),
        "test_ids_hash": nf.canonical_hash(test_ids),
        "counts": {"all": len(predictors), "development": len(dev_ids), "test": len(test_ids)},
        "test_target_fields_accessed": False,
        "validated_feature_schema": {"ssl": [len(predictors), *nf.SSL_SHAPE],
                                     "mfcc": [len(predictors), nf.MFCC_SIZE]},
    }
    write_json(output / "view_manifest.json", manifest)
    return {**result, "view_manifest_sha256": nf.sha256_file(output / "view_manifest.json")}


def load_prepared_inputs(input_dir: Path, config):
    try:
        manifest = json.loads((input_dir / "view_manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FinalInputError(f"cannot read prepared-input manifest: {exc}") from exc
    if (manifest.get("status") != "ok" or
            manifest.get("raw_config_sha256") != CONFIG_SHA256 or
            manifest.get("feature_source_fingerprint") !=
            config["expected_feature_source_fingerprint"] or
            manifest.get("test_target_fields_accessed") is not False):
        raise FinalInputError("prepared-input manifest is incompatible")
    for name, expected in manifest.get("prepared_file_sha256", {}).items():
        if nf.sha256_file(input_dir / name) != expected:
            raise FinalInputError(f"prepared input hash mismatch: {name}")
    if set(manifest.get("prepared_file_sha256", {})) != {
            "predictors.csv", "development_targets.csv", "validated_features.npz",
            "locked_config.json"}:
        raise FinalInputError("prepared input file inventory differs")
    if nf.sha256_file(input_dir / "locked_config.json") != CONFIG_SHA256:
        raise FinalInputError("prepared locked configuration differs")
    copied_npz_sha = nf.sha256_file(input_dir / "validated_features.npz")
    if copied_npz_sha != manifest["source_evidence_sha256"]["speaker_features.npz"]:
        raise FinalInputError("prepared feature NPZ bytes differ from source evidence")
    rebound_fingerprint = nf.canonical_hash({
        "config_fingerprint": manifest["feature_extraction_config_fingerprint"],
        "speaker_csv": manifest["source_evidence_sha256"]["speakers.csv"],
        "speaker_npz": copied_npz_sha,
    })
    if rebound_fingerprint != config["expected_feature_source_fingerprint"]:
        raise FinalInputError("prepared feature evidence fingerprint differs")
    predictors = read_exact_csv(input_dir / "predictors.csv", PREDICTOR_FIELDS)
    targets = read_exact_csv(input_dir / "development_targets.csv", DEVELOPMENT_TARGET_FIELDS)
    if len(predictors) != manifest["counts"]["all"] or len(targets) != manifest["counts"]["development"]:
        raise FinalInputError("prepared row counts differ from manifest")
    predictor_ids = [row["speaker_id"] for row in predictors]
    if (len(predictor_ids) != len(set(predictor_ids)) or
            nf.canonical_hash(predictor_ids) != manifest["speaker_order_hash"]):
        raise FinalInputError("prepared predictor IDs are duplicated or reordered")
    development = set(config["development_splits"])
    dev_ids = [row["speaker_id"] for row in predictors if row["split"] in development]
    test_ids = [row["speaker_id"] for row in predictors if row["split"] == "test"]
    if ([row["speaker_id"] for row in targets] != dev_ids or
            nf.canonical_hash(dev_ids) != manifest["development_ids_hash"] or
            nf.canonical_hash(test_ids) != manifest["test_ids_hash"] or
            len(dev_ids) != config["expected_development_speakers"] or
            len(test_ids) != config["expected_test_speakers"]):
        raise FinalInputError("prepared development/test ID coverage differs")
    target_splits = {row["speaker_id"]: row["split"] for row in targets}
    if any(target_splits.get(row["speaker_id"]) != row["split"]
           for row in predictors if row["split"] in development):
        raise FinalInputError("development target split differs from predictor split")
    ages, sexes = [], []
    allowed_languages = set(config["group_analysis"]["categories"]["language"])
    for row in predictors:
        try:
            age = float(row["age"])
        except ValueError as exc:
            raise FinalInputError("prepared age is invalid") from exc
        if (not np.isfinite(age) or row["sex"] not in {"Female", "Male"} or
                row["language"] not in allowed_languages or
                row["split"] not in development | {"test"}):
            raise FinalInputError("prepared predictor values are invalid")
        ages.append(age); sexes.append(0 if row["sex"] == "Female" else 1)
    target_values = {target: [] for target in config["targets"]}
    for row in targets:
        for target in config["targets"]:
            try:
                value = float(row[target])
            except ValueError as exc:
                raise FinalInputError("prepared development target is invalid") from exc
            if not np.isfinite(value):
                raise FinalInputError("prepared development target is non-finite")
            target_values[target].append(value)
    try:
        with np.load(input_dir / "validated_features.npz", allow_pickle=False) as values:
            if set(values.files) != NPZ_FIELDS:
                raise FinalInputError("prepared feature NPZ fields differ")
            ssl = values["ssl"].copy(); mfcc = values["mfcc"].copy()
            feature_ids = values["speaker_id"].astype("U32").copy()
            embedded_source = str(values["source_fingerprint"].item())
            embedded_metadata = str(values["speaker_metadata_sha256"].item())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise FinalInputError(f"cannot read prepared feature NPZ: {exc}") from exc
    if (ssl.shape != (len(predictors),) + nf.SSL_SHAPE or ssl.dtype != np.float32 or
            mfcc.shape != (len(predictors), nf.MFCC_SIZE) or mfcc.dtype != np.float32 or
            not np.isfinite(ssl).all() or not np.isfinite(mfcc).all() or
            not np.array_equal(feature_ids, np.asarray(predictor_ids, dtype="U32"))):
        raise FinalInputError("prepared feature arrays or ID order differ")
    if (embedded_source != manifest["feature_extraction_config_fingerprint"] or
            embedded_metadata != manifest["source_evidence_sha256"]["speakers.csv"]):
        raise FinalInputError("prepared feature NPZ is not bound to source evidence")
    return {
        "rows": predictors, "ids": predictor_ids, "dev_ids": dev_ids, "test_ids": test_ids,
        "age": np.asarray(ages), "sex": np.asarray(sexes),
        "targets": {key: np.asarray(value) for key, value in target_values.items()},
        "ssl": ssl, "mfcc": mfcc, "manifest": manifest,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan and args.output is None:
        parser.error("--output is required outside --plan")
    try:
        result = prepare(args.features, args.config, args.output, args.plan)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (FinalInputError, nf.FeatureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
