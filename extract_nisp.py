#!/usr/bin/env python3
"""Extract resumable frozen WavLM and MFCC features from NISP manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import nisp_features as nf


def _write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def make_plan(root: Path, limit: Optional[int], max_seconds: float,
              chunk_size: int) -> Dict[str, object]:
    rows, speaker_rows, manifest_fingerprint = nf.load_manifests(root)
    selected = nf.select_rows(rows, limit)
    config = nf.build_config(manifest_fingerprint, selected, limit, max_seconds, chunk_size)
    split_counts = {split: sum(row["split"] == split for row in selected)
                    for split in ("train", "val", "test")}
    language_counts = {prefix: sum(row["speaker_id"].startswith(prefix + "_") for row in selected)
                       for prefix in nf.PREFIXES}
    return {
        "rows": rows,
        "speaker_rows": speaker_rows,
        "selected": selected,
        "config": config,
        "summary": {
            "mode": "pilot" if limit is not None else "full",
            "selected_recordings": len(selected),
            "selected_speakers": len({row["speaker_id"] for row in selected}),
            "split_recordings": split_counts,
            "language_recordings": language_counts,
            "config_fingerprint": config["config_fingerprint"],
        },
    }


def _verify_or_write_config(output: Path, config: Mapping[str, object]) -> None:
    nf.validate_config(config)
    path = output / "config.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise nf.FeatureError(f"cannot read existing config: {exc}") from exc
        if existing != config:
            raise nf.FeatureError("output configuration fingerprint differs; use a separate output directory")
    else:
        nf.atomic_json(path, config)


def _load_runtime(model_dir: Path):
    if not os.environ.get("SLURM_JOB_ID"):
        raise nf.FeatureError("feature extraction requires a Slurm GPU job; use --plan on login nodes")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise nf.FeatureError("allocated job has no CUDA device")
    gpu_name = torch.cuda.get_device_name(0)
    if "H200" not in gpu_name:
        raise nf.FeatureError(f"expected an H200 allocation, found {gpu_name}")

    import soundfile as sf
    import scipy
    import torchaudio
    import transformers
    from scipy.signal import resample_poly
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    try:
        provenance = json.loads((model_dir / "download.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise nf.FeatureError(f"cannot read model provenance: {exc}") from exc
    if provenance.get("model_id") != nf.MODEL_ID or provenance.get("revision") != nf.REVISION:
        raise nf.FeatureError("model provenance does not match the fixed model revision")

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model_dir), local_files_only=True)
    model = WavLMModel.from_pretrained(
        str(model_dir), local_files_only=True, output_hidden_states=True
    ).to("cuda").eval()
    if model.config.hidden_size != nf.SSL_SHAPE[1] or model.config.num_hidden_layers != nf.SSL_SHAPE[0]:
        raise nf.FeatureError("WavLM architecture does not match 12 x 768 feature contract")
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=nf.TARGET_RATE,
        n_mfcc=13,
        log_mels=True,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40, "power": 2.0},
    )
    return {
        "np": np, "torch": torch, "sf": sf, "resample_poly": resample_poly,
        "extractor": extractor, "model": model, "mfcc": mfcc, "gpu_name": gpu_name,
        "versions": {
            "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
            "torchaudio": torchaudio.__version__, "transformers": transformers.__version__,
            "scipy": scipy.__version__, "soundfile": sf.__version__,
        },
    }


def validate_hidden_state_contract(hidden_states, last_hidden_state, allclose) -> None:
    """Check WavLM embedding+12-layer alignment without selecting or searching layers."""
    if hidden_states is None or len(hidden_states) != 13:
        raise nf.FeatureError(
            f"expected embedding plus 12 hidden states, found {0 if hidden_states is None else len(hidden_states)}"
        )
    expected_shape = tuple(last_hidden_state.shape)
    if any(tuple(layer.shape) != expected_shape for layer in hidden_states):
        raise nf.FeatureError("WavLM hidden state shapes are not aligned")
    if not allclose(hidden_states[-1], last_hidden_state):
        raise nf.FeatureError("hidden_states[-1] does not match last_hidden_state")


def _extract_one(row: Mapping[str, str], max_seconds: float, runtime: Mapping[str, object]):
    np = runtime["np"]
    torch = runtime["torch"]
    sf = runtime["sf"]
    load_started = time.monotonic()
    with sf.SoundFile(row["audio_path"]) as audio:
        source_rate = int(audio.samplerate)
        if audio.channels != 1 or source_rate <= 0:
            raise nf.FeatureError(f"expected positive-rate mono audio: {row['audio_path']}")
        source = audio.read(min(audio.frames, int(max_seconds * source_rate)), dtype="float32")
    if source.size == 0 or not np.isfinite(source).all():
        raise nf.FeatureError(f"empty or non-finite audio: {row['audio_path']}")
    divisor = math.gcd(source_rate, nf.TARGET_RATE)
    samples = runtime["resample_poly"](
        source, nf.TARGET_RATE // divisor, source_rate // divisor
    ).astype(np.float32)
    expected = (len(source) * nf.TARGET_RATE + source_rate - 1) // source_rate
    if len(samples) != expected or len(samples) == 0 or not np.isfinite(samples).all():
        raise nf.FeatureError(f"resampling validation failed: {row['audio_path']}")
    load_seconds = time.monotonic() - load_started

    extraction_started = time.monotonic()
    inputs = runtime["extractor"](
        samples, sampling_rate=nf.TARGET_RATE, return_tensors="pt", padding=False,
        return_attention_mask=False,
    )
    with torch.inference_mode():
        output = runtime["model"](
            input_values=inputs.input_values.to("cuda"), output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states
        validate_hidden_state_contract(
            hidden, output.last_hidden_state,
            lambda left, right: torch.allclose(left, right, rtol=1e-5, atol=1e-6),
        )
        ssl = torch.stack([layer.mean(dim=1)[0] for layer in hidden[1:]], dim=0)
        ssl = ssl.float().cpu().numpy()
        waveform = torch.from_numpy(samples).unsqueeze(0)
        coefficients = runtime["mfcc"](waveform)[0]
        traditional = torch.cat(
            (coefficients.mean(dim=1), coefficients.std(dim=1, unbiased=False)), dim=0
        ).float().cpu().numpy()
    ssl = nf.validate_layer_stack(ssl)
    traditional = np.asarray(traditional, dtype=np.float32)
    if traditional.shape != (nf.MFCC_SIZE,) or not np.isfinite(traditional).all():
        raise nf.FeatureError(f"invalid MFCC features: {row['audio_path']}")
    extract_seconds = time.monotonic() - extraction_started
    return {
        "ssl": ssl,
        "mfcc": traditional,
        "source_rate": source_rate,
        "source_frames": len(source),
        "target_frames": len(samples),
        "feature_frames": int(output.last_hidden_state.shape[1]),
        "load_seconds": load_seconds,
        "extract_seconds": extract_seconds,
    }


def _assemble(output: Path, selected: Sequence[Mapping[str, str]], config: Mapping[str, object],
              full_speakers: Sequence[Mapping[str, str]], chunk_size: int) -> Dict[str, object]:
    import numpy as np

    chunks = []
    for chunk_number, start, end in nf.chunk_ranges(len(selected), chunk_size):
        chunks.append(nf.validate_chunk(output / "chunks" / f"chunk-{chunk_number:06d}.npz",
                                        selected[start:end]))
    source_index = np.concatenate([item["source_index"] for item in chunks])
    source_key = np.concatenate([item["source_key"] for item in chunks]).astype("U64")
    ssl = np.concatenate([item["ssl"] for item in chunks]).astype(np.float32, copy=False)
    mfcc = np.concatenate([item["mfcc"] for item in chunks]).astype(np.float32, copy=False)
    source_rate = np.concatenate([item["source_rate"] for item in chunks])
    source_frames = np.concatenate([item["source_frames"] for item in chunks])
    target_frames = np.concatenate([item["target_frames"] for item in chunks])
    feature_frames = np.concatenate([item["feature_frames"] for item in chunks])
    if len(source_index) != len(selected):
        raise nf.FeatureError("assembled checkpoint coverage differs from selection")
    nf.atomic_npz(output / "utterance_features.npz", source_index=source_index,
                  source_key=source_key, ssl=ssl, mfcc=mfcc, source_rate=source_rate,
                  source_frames=source_frames, target_frames=target_frames,
                  feature_frames=feature_frames)

    utterance_fields = list(nf.MANIFEST_COLUMNS) + ["source_index", "source_key", "source_rate",
                                                        "source_frames", "target_frames", "feature_frames"]
    utterance_rows = []
    for index, row in enumerate(selected):
        item = dict(row)
        item.update(source_rate=int(source_rate[index]), source_frames=int(source_frames[index]),
                    target_frames=int(target_frames[index]), feature_frames=int(feature_frames[index]))
        utterance_rows.append(item)
    _write_csv_atomic(output / "utterances.csv", utterance_fields, utterance_rows)
    speaker_rows, speaker_ssl, speaker_mfcc = nf.aggregate_speakers(selected, ssl, mfcc)
    if config["limit"] is None:
        expected = {row["speaker_id"] for row in full_speakers}
        observed = {row["speaker_id"] for row in speaker_rows}
        if observed != expected:
            raise nf.FeatureError("full extraction does not cover every manifest speaker")
    speaker_fields = ["speaker_id", "split", "height_cm", "weight_kg", "bmi", "age",
                      "sex", "language", "num_recordings"]
    _write_csv_atomic(output / "speakers.csv", speaker_fields, speaker_rows)
    speaker_metadata_sha256 = nf.sha256_file(output / "speakers.csv")
    speaker_ids = np.asarray([row["speaker_id"] for row in speaker_rows], dtype="U32")
    nf.atomic_npz(
        output / "speaker_features.npz", ssl=speaker_ssl, mfcc=speaker_mfcc,
        speaker_id=speaker_ids,
        source_fingerprint=np.asarray(config["config_fingerprint"], dtype="U64"),
        speaker_metadata_sha256=np.asarray(speaker_metadata_sha256, dtype="U64"),
    )
    return {
        "utterance_shape": list(ssl.shape),
        "speaker_shape": list(speaker_ssl.shape),
        "speakers": len(speaker_rows),
        "speaker_metadata_sha256": speaker_metadata_sha256,
    }


def run_extraction(args: argparse.Namespace, plan: Mapping[str, object]) -> Dict[str, object]:
    import fcntl
    import numpy as np

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock_handle = (output / ".extract.lock").open("a+")
    try:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise nf.FeatureError("another extraction process holds this output lock") from exc
        _verify_or_write_config(output, plan["config"])
        selected = plan["selected"]
        runtime = _load_runtime(args.model_dir.resolve())
        metadata: Dict[str, object] = {
            "status": "processing", "config_fingerprint": plan["config"]["config_fingerprint"],
            "job_id": os.environ["SLURM_JOB_ID"], "gpu": runtime["gpu_name"],
            "selected": len(selected), "attempted": 0, "success": 0, "errors": [],
            "computed_this_attempt": 0, "reused_records": 0,
            "load_seconds": 0.0, "extract_seconds": 0.0,
            "versions": runtime["versions"],
        }
        nf.atomic_json(output / "metadata.json", metadata)
        completed = 0
        for chunk_number, start, end in nf.chunk_ranges(len(selected), args.chunk_size):
            chunk_rows = selected[start:end]
            chunk_path = output / "chunks" / f"chunk-{chunk_number:06d}.npz"
            try:
                nf.validate_chunk(chunk_path, chunk_rows)
                completed += len(chunk_rows)
                metadata["reused_records"] = int(metadata["reused_records"]) + len(chunk_rows)
                continue
            except nf.FeatureError:
                pass
            values: Dict[str, List[object]] = {
                "ssl": [], "mfcc": [], "source_rate": [], "source_frames": [],
                "target_frames": [], "feature_frames": [],
            }
            try:
                for row in chunk_rows:
                    metadata["attempted"] = int(metadata["attempted"]) + 1
                    result = _extract_one(row, args.max_seconds, runtime)
                    for field in values:
                        values[field].append(result[field])
                    metadata["load_seconds"] = float(metadata["load_seconds"]) + result["load_seconds"]
                    metadata["extract_seconds"] = float(metadata["extract_seconds"]) + result["extract_seconds"]
                    metadata["success"] = int(metadata["success"]) + 1
                    metadata["computed_this_attempt"] = int(metadata["computed_this_attempt"]) + 1
                    print(f"finished {start + len(values['ssl'])}/{len(selected)} {row['speaker_id']}", flush=True)
            except Exception as exc:
                metadata["status"] = "incomplete"
                metadata["errors"].append({
                    "source_index": int(row["source_index"]), "source_key": row["source_key"],
                    "audio_path": row["audio_path"], "error": str(exc),
                })
                nf.atomic_json(output / "metadata.json", metadata)
                _append_jsonl(output / "attempts.jsonl", {
                    "status": "incomplete", "job_id": metadata["job_id"],
                    "config_fingerprint": metadata["config_fingerprint"],
                    "attempted": metadata["attempted"], "success": metadata["success"],
                    "computed_this_attempt": metadata["computed_this_attempt"],
                    "reused_records": metadata["reused_records"],
                    "errors": metadata["errors"],
                })
                raise
            nf.atomic_npz(
                chunk_path,
                source_index=np.asarray([int(row["source_index"]) for row in chunk_rows], dtype=np.int64),
                source_key=np.asarray([row["source_key"] for row in chunk_rows], dtype="U64"),
                ssl=np.stack(values["ssl"]).astype(np.float32),
                mfcc=np.stack(values["mfcc"]).astype(np.float32),
                source_rate=np.asarray(values["source_rate"], dtype=np.int32),
                source_frames=np.asarray(values["source_frames"], dtype=np.int64),
                target_frames=np.asarray(values["target_frames"], dtype=np.int64),
                feature_frames=np.asarray(values["feature_frames"], dtype=np.int64),
            )
            nf.validate_chunk(chunk_path, chunk_rows)
            completed += len(chunk_rows)
            metadata["completed"] = completed
            nf.atomic_json(output / "metadata.json", metadata)
        try:
            assembled = _assemble(
                output, selected, plan["config"], plan["speaker_rows"], args.chunk_size
            )
        except BaseException as exc:
            metadata["status"] = "incomplete"
            metadata["errors"] = [{"stage": "assemble", "error": str(exc)}]
            nf.atomic_json(output / "metadata.json", metadata)
            _append_jsonl(output / "attempts.jsonl", {
                "status": "incomplete", "job_id": metadata["job_id"],
                "config_fingerprint": metadata["config_fingerprint"],
                "attempted": metadata["attempted"], "success": metadata["success"],
                "computed_this_attempt": metadata["computed_this_attempt"],
                "reused_records": metadata["reused_records"],
                "errors": metadata["errors"],
            })
            raise
        metadata.update(assembled)
        metadata.update(
            status="ok", completed=len(selected), success=len(selected), errors=[],
            peak_gpu_memory_bytes=runtime["torch"].cuda.max_memory_allocated(),
            torch_version=runtime["torch"].__version__, cuda_version=runtime["torch"].version.cuda,
        )
        nf.atomic_json(output / "metadata.json", metadata)
        _append_jsonl(output / "attempts.jsonl", {
            "status": "ok", "job_id": metadata["job_id"],
            "config_fingerprint": metadata["config_fingerprint"],
            "attempted": metadata["attempted"], "success": metadata["success"], "errors": [],
            "computed_this_attempt": metadata["computed_this_attempt"],
            "reused_records": metadata["reused_records"],
        })
        return metadata
    finally:
        lock_handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nisp"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-seconds", type=float, default=nf.MAX_SECONDS)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--plan", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_seconds) or args.max_seconds != nf.MAX_SECONDS:
        parser.error("the first-batch protocol fixes --max-seconds at 10")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be positive")
    try:
        plan = make_plan(args.root, args.limit, args.max_seconds, args.chunk_size)
        if args.plan:
            print(json.dumps(plan["summary"], ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if args.model_dir is None or args.output is None:
            parser.error("--model-dir and --output are required for extraction")
        result = run_extraction(args, plan)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (nf.FeatureError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
