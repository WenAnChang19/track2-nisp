#!/usr/bin/env python3
"""Data contracts and checkpoint helpers for frozen NISP speech features."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

MODEL_ID = "microsoft/wavlm-base-plus"
REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
TARGET_RATE = 16000
MAX_SECONDS = 10.0
PREFIXES = ("Hin", "Kan", "Mal", "Tam", "Tel")
SSL_SHAPE = (12, 768)
MFCC_SIZE = 26
FEATURE_DEFINITION = {
    "ssl": "WavLM transformer hidden_states[1:13], time mean, float32",
    "mfcc": "torchaudio MFCC13 log_mels=True mean+population_std on same samples",
    "mfcc_parameters": {
        "sample_rate": TARGET_RATE,
        "n_mfcc": 13,
        "n_fft": 400,
        "hop_length": 160,
        "n_mels": 40,
        "power": 2.0,
    },
}
MANIFEST_COLUMNS = (
    "audio_path", "speaker_id", "split", "height_cm", "weight_kg", "bmi", "age", "sex"
)
LABEL_COLUMNS = ("height_cm", "weight_kg", "bmi", "age", "sex")


class FeatureError(RuntimeError):
    """Raised when an input or saved feature violates the experiment contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise FeatureError(f"empty CSV: {path}")
            return [dict(row) for row in reader]
    except OSError as exc:
        raise FeatureError(f"cannot read {path}: {exc}") from exc


def _validate_number(raw: str, field: str, speaker_id: str) -> None:
    try:
        value = float(raw)
    except ValueError as exc:
        raise FeatureError(f"{speaker_id}: invalid {field}") from exc
    if not math.isfinite(value):
        raise FeatureError(f"{speaker_id}: non-finite {field}")


def load_manifests(root: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], str]:
    """Load and cross-check audit, split, utterance and speaker manifests.

    This intentionally reads manifest files only; it does not stat or decode every WAV.
    """
    root = Path(root).resolve()
    manifest_dir = root / "manifests"
    names = ("audit.json", "all.csv", "train.csv", "val.csv", "test.csv", "speakers.csv")
    try:
        audit = json.loads((manifest_dir / "audit.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeatureError(f"cannot read manifest audit: {exc}") from exc
    if audit.get("status") != "ok" or audit.get("manifests_written") is not True:
        raise FeatureError("manifest audit is stale or incomplete")

    rows = read_csv(manifest_dir / "all.csv")
    if not rows:
        raise FeatureError("all.csv has no recordings")
    for row in rows:
        missing = [field for field in MANIFEST_COLUMNS if field not in row]
        if missing:
            raise FeatureError(f"all.csv missing columns: {missing}")

    path_keys: Dict[str, Dict[str, str]] = {}
    speaker_state: Dict[str, Tuple[str, ...]] = {}
    split_speakers: Dict[str, set] = {name: set() for name in ("train", "val", "test")}
    for source_index, row in enumerate(rows):
        split = row["split"]
        if split not in split_speakers:
            raise FeatureError(f"invalid split at all.csv row {source_index}: {split!r}")
        raw_path = Path(row["audio_path"])
        if not raw_path.is_absolute() or ".." in raw_path.parts:
            raise FeatureError(f"audio path must be normalized and absolute: {raw_path}")
        audio_path = os.path.normpath(str(raw_path))
        try:
            Path(audio_path).relative_to(root / "audio")
        except ValueError as exc:
            raise FeatureError(f"audio path is outside root/audio: {audio_path}") from exc
        if audio_path in path_keys:
            raise FeatureError(f"duplicate audio path: {audio_path}")
        row["audio_path"] = audio_path
        if not Path(audio_path).name.startswith(row["speaker_id"] + "_Eng_"):
            raise FeatureError(f"audio filename and speaker ID disagree: {audio_path}")
        row["source_index"] = str(source_index)
        row["source_key"] = canonical_hash({field: row[field] for field in MANIFEST_COLUMNS})
        path_keys[audio_path] = row
        speaker_id = row["speaker_id"]
        values = tuple(row[field] for field in ("split",) + LABEL_COLUMNS)
        if speaker_id in speaker_state and speaker_state[speaker_id] != values:
            raise FeatureError(f"mixed labels or splits for speaker {speaker_id}")
        speaker_state[speaker_id] = values
        split_speakers[split].add(speaker_id)
        for field in ("height_cm", "weight_kg", "bmi", "age"):
            _validate_number(row[field], field, speaker_id)
        if row["sex"] not in {"Female", "Male"}:
            raise FeatureError(f"{speaker_id}: unsupported sex {row['sex']!r}")

    if any(split_speakers[a] & split_speakers[b]
           for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise FeatureError("speaker appears in more than one split")

    combined: Dict[str, Dict[str, str]] = {}
    for split in ("train", "val", "test"):
        split_rows = read_csv(manifest_dir / f"{split}.csv")
        for row in split_rows:
            if row.get("split") != split:
                raise FeatureError(f"non-{split} row in {split}.csv")
            raw_path = Path(row["audio_path"])
            if not raw_path.is_absolute() or ".." in raw_path.parts:
                raise FeatureError(f"audio path must be normalized and absolute: {raw_path}")
            key = os.path.normpath(str(raw_path))
            if key in combined:
                raise FeatureError(f"recording appears in multiple split CSVs: {key}")
            combined[key] = row
    if set(combined) != set(path_keys):
        raise FeatureError("all.csv and split CSV coverage differ")
    for key, all_row in path_keys.items():
        if any(combined[key].get(field) != all_row[field] for field in MANIFEST_COLUMNS):
            raise FeatureError(f"all.csv and split CSV disagree: {key}")

    speaker_rows = read_csv(manifest_dir / "speakers.csv")
    speaker_by_id: Dict[str, Dict[str, str]] = {}
    for row in speaker_rows:
        speaker_id = row.get("speaker_id", "")
        if not speaker_id or speaker_id in speaker_by_id:
            raise FeatureError(f"duplicate or empty speaker ID in speakers.csv: {speaker_id!r}")
        speaker_by_id[speaker_id] = row
    if set(speaker_by_id) != set(speaker_state):
        raise FeatureError("speakers.csv and recording speakers differ")
    for speaker_id, state in speaker_state.items():
        row = speaker_by_id[speaker_id]
        expected = dict(zip(("split",) + LABEL_COLUMNS, state))
        if any(row.get(field) != value for field, value in expected.items()):
            raise FeatureError(f"speakers.csv labels disagree for {speaker_id}")
        expected_count = sum(item["speaker_id"] == speaker_id for item in rows)
        if int(row.get("num_recordings", "-1")) != expected_count:
            raise FeatureError(f"speakers.csv recording count disagrees for {speaker_id}")

    counts = audit.get("counts_by_split", {})
    for split in ("train", "val", "test"):
        observed = sum(row["split"] == split for row in rows)
        if counts.get(split, {}).get("recordings") != observed:
            raise FeatureError(f"audit recording count is stale for {split}")

    fingerprint = canonical_hash({name: sha256_file(manifest_dir / name) for name in names})
    return rows, speaker_rows, fingerprint


def select_rows(rows: Sequence[Mapping[str, str]], limit: Optional[int]) -> List[Dict[str, str]]:
    """Select all rows or a deterministic balanced training-only pilot."""
    if limit is None:
        return [dict(row) for row in rows]
    if limit <= 0:
        raise FeatureError("--limit must be positive")
    groups: Dict[str, List[Mapping[str, str]]] = {prefix: [] for prefix in PREFIXES}
    for row in rows:
        if row["split"] != "train":
            continue
        prefix = row["speaker_id"].split("_", 1)[0]
        if prefix in groups:
            groups[prefix].append(row)
    for prefix in PREFIXES:
        groups[prefix].sort(
            key=lambda item: hashlib.sha256(
                ("42\0" + item["audio_path"]).encode("utf-8")
            ).hexdigest()
        )
    selected: List[Dict[str, str]] = []
    offset = 0
    while len(selected) < limit:
        advanced = False
        for prefix in PREFIXES:
            if offset < len(groups[prefix]) and len(selected) < limit:
                selected.append(dict(groups[prefix][offset]))
                advanced = True
        if not advanced:
            break
        offset += 1
    if len(selected) != limit:
        raise FeatureError(f"requested {limit} pilot rows but only {len(selected)} are available")
    return selected


def build_config(manifest_fingerprint: str, selected: Sequence[Mapping[str, str]],
                 limit: Optional[int], max_seconds: float, chunk_size: int) -> Dict[str, object]:
    inventory = [
        {"source_index": int(row["source_index"]), "source_key": row["source_key"],
         "audio_path": row["audio_path"]}
        for row in selected
    ]
    core: Dict[str, object] = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "target_rate": TARGET_RATE,
        "max_seconds": max_seconds,
        "chunk_size": chunk_size,
        "limit": limit,
        "feature_definition": FEATURE_DEFINITION,
        "manifest_fingerprint": manifest_fingerprint,
        "selection_inventory": inventory,
        "selection_fingerprint": canonical_hash(inventory),
    }
    core["config_fingerprint"] = canonical_hash(core)
    return core


def validate_config(config: Mapping[str, object]) -> None:
    claimed = config.get("config_fingerprint")
    payload = dict(config)
    payload.pop("config_fingerprint", None)
    if not isinstance(claimed, str) or claimed != canonical_hash(payload):
        raise FeatureError("configuration fingerprint is invalid")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_npz(path: Path, **arrays: object) -> None:
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def validate_chunk(path: Path, expected_rows: Sequence[Mapping[str, str]]) -> Dict[str, object]:
    import numpy as np
    try:
        with path.open("rb") as handle:
            with np.load(handle, allow_pickle=False) as data:
                required = {"source_index", "source_key", "ssl", "mfcc", "source_rate",
                            "source_frames", "target_frames", "feature_frames"}
                if set(data.files) != required:
                    raise FeatureError(f"checkpoint fields differ: {path}")
                arrays = {name: data[name].copy() for name in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise FeatureError(f"cannot validate checkpoint {path}: {exc}") from exc
    count = len(expected_rows)
    expected_indices = np.asarray([int(row["source_index"]) for row in expected_rows], dtype=np.int64)
    expected_keys = np.asarray([row["source_key"] for row in expected_rows], dtype="U64")
    if not np.array_equal(arrays["source_index"], expected_indices):
        raise FeatureError(f"checkpoint indices differ: {path}")
    if not np.array_equal(arrays["source_key"].astype("U64"), expected_keys):
        raise FeatureError(f"checkpoint source keys differ: {path}")
    if arrays["ssl"].shape != (count,) + SSL_SHAPE or arrays["ssl"].dtype != np.float32:
        raise FeatureError(f"invalid SSL checkpoint shape or dtype: {path}")
    if arrays["mfcc"].shape != (count, MFCC_SIZE) or arrays["mfcc"].dtype != np.float32:
        raise FeatureError(f"invalid MFCC checkpoint shape or dtype: {path}")
    if not np.isfinite(arrays["ssl"]).all() or not np.isfinite(arrays["mfcc"]).all():
        raise FeatureError(f"non-finite checkpoint values: {path}")
    for field in ("source_rate", "source_frames", "target_frames", "feature_frames"):
        if arrays[field].shape != (count,) or np.any(arrays[field] <= 0):
            raise FeatureError(f"invalid {field} in checkpoint: {path}")
    return arrays


def validate_layer_stack(value: object) -> object:
    """Validate one utterance's pooled 12-layer WavLM representation."""
    import numpy as np
    array = np.asarray(value, dtype=np.float32)
    if array.shape != SSL_SHAPE:
        raise FeatureError(f"expected pooled SSL shape {SSL_SHAPE}, found {array.shape}")
    if not np.isfinite(array).all():
        raise FeatureError("pooled SSL features contain NaN or Inf")
    return array


def aggregate_speakers(rows: Sequence[Mapping[str, str]], ssl: np.ndarray,
                       mfcc: np.ndarray) -> Tuple[List[Dict[str, str]], np.ndarray, np.ndarray]:
    """Average utterance vectors equally within each speaker."""
    import numpy as np
    if len(rows) != len(ssl) or len(rows) != len(mfcc):
        raise FeatureError("row and feature counts differ")
    grouped: Dict[str, List[int]] = defaultdict(list)
    state: Dict[str, Tuple[str, ...]] = {}
    for index, row in enumerate(rows):
        speaker_id = row["speaker_id"]
        values = tuple(row[field] for field in ("split",) + LABEL_COLUMNS)
        if speaker_id in state and state[speaker_id] != values:
            raise FeatureError(f"mixed labels or splits for speaker {speaker_id}")
        state[speaker_id] = values
        grouped[speaker_id].append(index)
    metadata: List[Dict[str, str]] = []
    ssl_rows: List[np.ndarray] = []
    mfcc_rows: List[np.ndarray] = []
    for speaker_id in sorted(grouped):
        indices = grouped[speaker_id]
        split, height, weight, bmi, age, sex = state[speaker_id]
        metadata.append({
            "speaker_id": speaker_id, "split": split, "height_cm": height,
            "weight_kg": weight, "bmi": bmi, "age": age, "sex": sex,
            "language": speaker_id.split("_", 1)[0], "num_recordings": str(len(indices)),
        })
        ssl_rows.append(np.asarray(ssl[indices], dtype=np.float64).mean(axis=0).astype(np.float32))
        mfcc_rows.append(np.asarray(mfcc[indices], dtype=np.float64).mean(axis=0).astype(np.float32))
    return metadata, np.stack(ssl_rows), np.stack(mfcc_rows)


def chunk_ranges(count: int, chunk_size: int) -> Iterable[Tuple[int, int, int]]:
    for start in range(0, count, chunk_size):
        yield start // chunk_size, start, min(start + chunk_size, count)
