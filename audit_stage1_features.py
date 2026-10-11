#!/usr/bin/env python3
"""Independently check feature coverage, resampling and speaker aggregation."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def audit(directory):
    metadata = json.loads((directory / "metadata.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    rows = read_rows(directory / "utterances.csv")
    speakers = read_rows(directory / "speakers.csv")
    if metadata["status"] != "ok" or len(rows) != metadata["completed"]:
        raise ValueError("Incomplete extraction")
    with np.load(directory / "utterance_features.npz", allow_pickle=False) as data:
        ssl = data["ssl"]
        mfcc = data["mfcc"]
        if ssl.shape != (len(rows), 12, 768) or mfcc.shape != (len(rows), 26):
            raise ValueError("Unexpected utterance shape")
        if ssl.dtype != np.float32 or mfcc.dtype != np.float32:
            raise ValueError("Unexpected dtype")
        if not np.isfinite(ssl).all() or not np.isfinite(mfcc).all():
            raise ValueError("Nonfinite features")
        source_index = data["source_index"]
        source_key = data["source_key"]
        if source_index.shape != (len(rows),) or source_key.shape != (len(rows),):
            raise ValueError("Unexpected source inventory shape")
        for index, row in enumerate(rows):
            if int(row["source_index"]) != source_index[index]:
                raise ValueError("Source index mismatch")
            if row["source_key"] != source_key[index]:
                raise ValueError("Source key mismatch")
            rate, frames = int(row["source_rate"]), int(row["source_frames"])
            target = (frames * 16000 + rate - 1) // rate
            if target != int(row["target_frames"]) or frames > rate * 10 or frames <= 0:
                raise ValueError("Resampling or truncation mismatch")
        ids, inverse = np.unique([row["speaker_id"] for row in rows], return_inverse=True)
        counts = np.bincount(inverse)
        ssl_sum = np.zeros((len(ids), 12, 768), dtype=np.float64)
        mfcc_sum = np.zeros((len(ids), 26), dtype=np.float64)
        np.add.at(ssl_sum, inverse, ssl)
        np.add.at(mfcc_sum, inverse, mfcc)
        expected_ssl = (ssl_sum / counts[:, None, None]).astype(np.float32)
        expected_mfcc = (mfcc_sum / counts[:, None]).astype(np.float32)
    if list(ids) != [row["speaker_id"] for row in speakers]:
        raise ValueError("Speaker order mismatch")
    if list(counts) != [int(row["num_recordings"]) for row in speakers]:
        raise ValueError("Speaker recording counts mismatch")
    if len(rows) != len(config["selection_inventory"]):
        raise ValueError("Selection coverage mismatch")
    with np.load(directory / "speaker_features.npz", allow_pickle=False) as data:
        if not np.array_equal(data["speaker_id"], ids):
            raise ValueError("Speaker feature IDs mismatch")
        if not np.allclose(data["ssl"], expected_ssl, rtol=1e-6, atol=1e-6):
            raise ValueError("Independent SSL average differs")
        if not np.allclose(data["mfcc"], expected_mfcc, rtol=1e-6, atol=1e-6):
            raise ValueError("Independent MFCC average differs")
        digest = hashlib.sha256((directory / "speakers.csv").read_bytes()).hexdigest()
        if digest != data["speaker_metadata_sha256"].item() or digest != metadata["speaker_metadata_sha256"]:
            raise ValueError("Speaker metadata integrity mismatch")
    result = {
        "status": "ok", "utterances": len(rows), "speakers": len(speakers),
        "utterance_shape": [len(rows), 12, 768],
        "speaker_shape": [len(speakers), 12, 768],
        "speaker_split_counts": dict(Counter(row["split"] for row in speakers)),
        "recording_split_counts": dict(Counter(row["split"] for row in rows)),
        "all_finite": True, "resampling_lengths_verified": True,
        "independent_speaker_average_verified": True,
        "test_metrics_computed": False,
    }
    (directory / "independent-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.features), indent=2))
