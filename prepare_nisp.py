#!/usr/bin/env python3
"""Audit NISP metadata and build deterministic speaker-independent manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_ROOT = Path("data/nisp")
EXPECTED_HEADER = [
    "Speaker_ID",
    "Gender",
    "Mother_Tongue",
    "Height",
    "Shoulder_size",
    "Waist_size",
    "Weight",
    "Age",
    "Native_State",
    "Native_District",
]
MANIFEST_FIELDS = [
    "audio_path",
    "speaker_id",
    "split",
    "height_cm",
    "weight_kg",
    "bmi",
    "age",
    "sex",
]
SPEAKER_FIELDS = [
    "speaker_id",
    "official_split",
    "split",
    "height_cm",
    "weight_kg",
    "bmi",
    "age",
    "sex",
    "num_recordings",
]
SPEAKER_ID_RE = re.compile(r"^[A-Za-z]+_[0-9]+$")
ENGLISH_WAV_RE = re.compile(
    r"^(?P<speaker>[A-Za-z]+_[0-9]+)_Eng_(?P<utterance>.+)\.[Ww][Aa][Vv]$"
)


class PreparationError(RuntimeError):
    """Raised when an input invariant makes manifest generation unsafe."""


@dataclass(frozen=True)
class Speaker:
    speaker_id: str
    sex: str
    height_cm: Decimal
    weight_kg: Decimal
    age: Decimal

    @property
    def bmi(self) -> Decimal:
        return self.weight_kg / ((self.height_cm / Decimal("100")) ** 2)


@dataclass(frozen=True)
class Recording:
    path: Path
    speaker_id: str


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_positive_decimal(raw: str, field: str, line_number: int) -> Decimal:
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise PreparationError(
            f"total_spkrinfo.list line {line_number}: {field} is not numeric: {raw!r}"
        ) from exc
    if not value.is_finite() or value <= 0:
        raise PreparationError(
            f"total_spkrinfo.list line {line_number}: {field} must be positive"
        )
    return value


def read_metadata(path: Path) -> Dict[str, Speaker]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PreparationError(f"cannot read metadata file {path}: {exc}") from exc
    if not lines or lines[0].split() != EXPECTED_HEADER:
        found = lines[0].split() if lines else []
        raise PreparationError(
            f"unexpected total_spkrinfo.list header: expected {EXPECTED_HEADER}, found {found}"
        )

    speakers: Dict[str, Speaker] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != len(EXPECTED_HEADER):
            raise PreparationError(
                f"total_spkrinfo.list line {line_number}: expected 10 fields, found {len(fields)}"
            )
        speaker_id, sex = fields[0], fields[1]
        if not SPEAKER_ID_RE.fullmatch(speaker_id):
            raise PreparationError(
                f"total_spkrinfo.list line {line_number}: invalid speaker ID {speaker_id!r}"
            )
        if speaker_id in speakers:
            raise PreparationError(f"duplicate metadata speaker ID: {speaker_id}")
        if sex not in {"Female", "Male"}:
            raise PreparationError(
                f"total_spkrinfo.list line {line_number}: unsupported Gender {sex!r}"
            )
        speakers[speaker_id] = Speaker(
            speaker_id=speaker_id,
            sex=sex,
            height_cm=parse_positive_decimal(fields[3], "Height", line_number),
            weight_kg=parse_positive_decimal(fields[6], "Weight", line_number),
            age=parse_positive_decimal(fields[7], "Age", line_number),
        )
    if not speakers:
        raise PreparationError("total_spkrinfo.list has no speaker rows")
    return speakers


def read_official_split(path: Path, suffix: str) -> List[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PreparationError(f"cannot read split file {path}: {exc}") from exc
    ids: List[str] = []
    expected_tail = "___" + suffix
    for line_number, line in enumerate(lines, start=1):
        token = line.strip()
        if not token or not token.endswith(expected_tail):
            raise PreparationError(
                f"{path.name} line {line_number}: expected SPEAKER{expected_tail}, found {token!r}"
            )
        speaker_id = token[: -len(expected_tail)]
        if not SPEAKER_ID_RE.fullmatch(speaker_id):
            raise PreparationError(
                f"{path.name} line {line_number}: invalid speaker ID {speaker_id!r}"
            )
        ids.append(speaker_id)
    duplicates = sorted(speaker_id for speaker_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise PreparationError(f"duplicate IDs in {path.name}: {duplicates}")
    if not ids:
        raise PreparationError(f"{path.name} is empty")
    return ids


def validate_official_splits(
    speakers: Mapping[str, Speaker], train_ids: Sequence[str], test_ids: Sequence[str]
) -> None:
    overlap = sorted(set(train_ids) & set(test_ids))
    if overlap:
        raise PreparationError(f"official train/test overlap: {overlap}")
    assigned = set(train_ids) | set(test_ids)
    metadata_ids = set(speakers)
    missing = sorted(metadata_ids - assigned)
    unknown = sorted(assigned - metadata_ids)
    if missing or unknown:
        raise PreparationError(
            "official split coverage mismatch: "
            f"metadata_without_split={missing}, split_without_metadata={unknown}"
        )


def choose_validation_speakers(
    speakers: Mapping[str, Speaker], train_ids: Sequence[str], fraction: float, seed: int
) -> List[str]:
    if not 0 < fraction < 1:
        raise PreparationError("--val-fraction must be greater than 0 and less than 1")
    by_sex: Dict[str, List[str]] = defaultdict(list)
    for speaker_id in sorted(train_ids):
        by_sex[speakers[speaker_id].sex].append(speaker_id)

    rng = random.Random(seed)
    selected: List[str] = []
    for sex in sorted(by_sex):
        candidates = by_sex[sex]
        rng.shuffle(candidates)
        count = int(len(candidates) * fraction + 0.5)
        selected.extend(candidates[:count])
    return sorted(selected)


def iter_wav_paths(audio_root: Path) -> Iterable[Path]:
    if not audio_root.is_dir():
        raise PreparationError(f"audio directory does not exist: {audio_root}")
    for directory, dirnames, filenames in os.walk(str(audio_root)):
        dirnames.sort()
        for filename in sorted(filenames):
            if filename.lower().endswith(".wav"):
                yield Path(directory) / filename


def scan_audio(
    root: Path, speakers: Mapping[str, Speaker]
) -> Tuple[List[Recording], Dict[str, object], str]:
    recordings: List[Recording] = []
    non_english: List[str] = []
    malformed_english: List[str] = []
    unknown: List[Tuple[str, str]] = []
    basename_paths: Dict[str, List[str]] = defaultdict(list)
    index_digest = hashlib.sha256()
    total_wav = 0

    for path in iter_wav_paths(root / "audio"):
        total_wav += 1
        relative = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise PreparationError(f"cannot stat audio file {path}: {exc}") from exc
        index_digest.update(f"{relative}\t{size}\n".encode("utf-8"))
        basename_paths[path.name].append(relative)
        match = ENGLISH_WAV_RE.fullmatch(path.name)
        if match is None:
            if "_Eng_" in path.name:
                malformed_english.append(relative)
            else:
                non_english.append(relative)
            continue
        speaker_id = match.group("speaker")
        if speaker_id not in speakers:
            unknown.append((speaker_id, relative))
            continue
        recordings.append(Recording(path=path.resolve(), speaker_id=speaker_id))

    duplicates = {
        basename: paths
        for basename, paths in sorted(basename_paths.items())
        if len(paths) > 1
    }
    details: Dict[str, object] = {
        "total_wav_files": total_wav,
        "english_known_speaker_recordings": len(recordings),
        "exclusions": {
            "non_english_wav": {
                "count": len(non_english),
                "examples": non_english[:20],
            }
        },
        "anomalies": {
            "unknown_audio_speakers": {
                "file_count": len(unknown),
                "speaker_ids": sorted({item[0] for item in unknown}),
                "examples": [item[1] for item in unknown[:20]],
            },
            "duplicate_utterance_filenames": {
                "count": len(duplicates),
                "examples": dict(list(duplicates.items())[:20]),
            },
            "malformed_english_filenames": {
                "count": len(malformed_english),
                "examples": malformed_english[:20],
            },
        },
    }
    return sorted(recordings, key=lambda item: item.path.as_posix()), details, index_digest.hexdigest()


def manifest_row(recording: Recording, speaker: Speaker, split: str) -> Dict[str, object]:
    return {
        "audio_path": recording.path.as_posix(),
        "speaker_id": speaker.speaker_id,
        "split": split,
        "height_cm": decimal_text(speaker.height_cm),
        "weight_kg": decimal_text(speaker.weight_kg),
        "bmi": format(speaker.bmi, ".6f"),
        "age": decimal_text(speaker.age),
        "sex": speaker.sex,
    }


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare(root: Path, val_fraction: float = 0.2, seed: int = 42) -> Dict[str, object]:
    root = root.resolve()
    metadata_dir = root / "metadata"
    metadata_path = metadata_dir / "total_spkrinfo.list"
    train_path = metadata_dir / "train_spkrID"
    test_path = metadata_dir / "test_spkrID"

    speakers = read_metadata(metadata_path)
    official_train = read_official_split(train_path, "TRN")
    official_test = read_official_split(test_path, "TST")
    validate_official_splits(speakers, official_train, official_test)
    validation = choose_validation_speakers(speakers, official_train, val_fraction, seed)
    validation_set = set(validation)
    derived = {
        "train": sorted(set(official_train) - validation_set),
        "val": validation,
        "test": sorted(official_test),
    }
    split_for = {
        speaker_id: split
        for split, ids in derived.items()
        for speaker_id in ids
    }

    recordings, scan, audio_index_hash = scan_audio(root, speakers)
    recording_counts = Counter(recording.speaker_id for recording in recordings)
    rows_by_split: Dict[str, List[Dict[str, object]]] = {name: [] for name in derived}
    for recording in recordings:
        speaker = speakers[recording.speaker_id]
        split = split_for[recording.speaker_id]
        rows_by_split[split].append(manifest_row(recording, speaker, split))
    all_rows = sorted(
        (row for rows in rows_by_split.values() for row in rows),
        key=lambda row: str(row["audio_path"]),
    )

    missing_by_split = {
        split: [speaker_id for speaker_id in ids if recording_counts[speaker_id] == 0]
        for split, ids in derived.items()
    }
    counts_by_split = {
        split: {
            "assigned_speakers": len(ids),
            "recorded_speakers": sum(recording_counts[speaker_id] > 0 for speaker_id in ids),
            "recordings": len(rows_by_split[split]),
        }
        for split, ids in derived.items()
    }
    empty_recorded_splits = sorted(
        split for split, counts in counts_by_split.items() if counts["recorded_speakers"] == 0
    )
    anomaly_data = scan["anomalies"]
    assert isinstance(anomaly_data, dict)
    anomaly_data["empty_recorded_splits"] = {
        "count": len(empty_recorded_splits),
        "splits": empty_recorded_splits,
    }
    unsafe = any(
        int(anomaly_data[name]["count" if name != "unknown_audio_speakers" else "file_count"]) > 0
        for name in (
            "unknown_audio_speakers",
            "duplicate_utterance_filenames",
            "malformed_english_filenames",
            "empty_recorded_splits",
        )
    )

    audit: Dict[str, object] = {
        "status": "error" if unsafe else "preparing",
        "manifests_written": False,
        "seed": seed,
        "validation_fraction": val_fraction,
        "inputs": {
            "total_spkrinfo.list": {
                "path": metadata_path.relative_to(root).as_posix(),
                "sha256": sha256_file(metadata_path),
            },
            "train_spkrID": {
                "path": train_path.relative_to(root).as_posix(),
                "sha256": sha256_file(train_path),
            },
            "test_spkrID": {
                "path": test_path.relative_to(root).as_posix(),
                "sha256": sha256_file(test_path),
            },
            "audio_index": {
                "sha256": audio_index_hash,
                "definition": "SHA-256 of sorted root-relative WAV path, tab, byte size, newline",
            },
        },
        "observed_vs_official": {
            "metadata_speakers": len(speakers),
            "official_train_speakers": len(official_train),
            "official_test_speakers": len(official_test),
            "recorded_speakers": len(recording_counts),
            "wav_files_observed": scan["total_wav_files"],
            "english_recordings_included": len(recordings),
        },
        "counts_by_split": counts_by_split,
        "speakers_with_no_recordings": missing_by_split,
        "official_split_assignments": {
            "train": sorted(official_train),
            "test": sorted(official_test),
        },
        "derived_split_assignments": derived,
        "exclusions": scan["exclusions"],
        "anomalies": anomaly_data,
    }

    output_dir = root / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "audit.json", audit)
    if unsafe:
        raise PreparationError(
            "unsafe audio mismatch; manifests were not written and existing CSV outputs may be stale "
            "(see manifests/audit.json)"
        )

    for split in ("train", "val", "test"):
        write_csv(output_dir / f"{split}.csv", MANIFEST_FIELDS, rows_by_split[split])
    write_csv(output_dir / "all.csv", MANIFEST_FIELDS, all_rows)

    speaker_rows: List[Dict[str, object]] = []
    official_train_set = set(official_train)
    for speaker_id in sorted(recording_counts):
        speaker = speakers[speaker_id]
        speaker_rows.append(
            {
                "speaker_id": speaker_id,
                "official_split": "train" if speaker_id in official_train_set else "test",
                "split": split_for[speaker_id],
                "height_cm": decimal_text(speaker.height_cm),
                "weight_kg": decimal_text(speaker.weight_kg),
                "bmi": format(speaker.bmi, ".6f"),
                "age": decimal_text(speaker.age),
                "sex": speaker.sex,
                "num_recordings": recording_counts[speaker_id],
            }
        )
    write_csv(output_dir / "speakers.csv", SPEAKER_FIELDS, speaker_rows)
    audit["status"] = "ok"
    audit["manifests_written"] = True
    write_json(output_dir / "audit.json", audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit NISP inputs and create speaker-independent English manifests."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="NISP dataset root")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        audit = prepare(args.root, args.val_fraction, args.seed)
    except (PreparationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    observed = audit["observed_vs_official"]
    print(
        f"wrote {args.root.resolve() / 'manifests'}: "
        f"{observed['english_recordings_included']} recordings, "
        f"{observed['recorded_speakers']} recorded speakers"
    )
    counts_by_split = audit["counts_by_split"]
    missing_by_split = audit["speakers_with_no_recordings"]
    for split in ("train", "val", "test"):
        counts = counts_by_split[split]
        print(
            f"{split}: {counts['recorded_speakers']}/{counts['assigned_speakers']} speakers, "
            f"{counts['recordings']} recordings, {len(missing_by_split[split])} missing speakers"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
