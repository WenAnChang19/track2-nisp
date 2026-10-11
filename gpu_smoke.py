#!/usr/bin/env python3
"""Five training clips only; all model execution requires a Slurm GPU job."""
import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

MODEL_ID = "microsoft/wavlm-base-plus"
REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
PREFIXES = ("Hin", "Kan", "Mal", "Tam", "Tel")
TARGET_RATE = 16000


def select_clips(root):
    root = Path(root).resolve()
    audit = json.loads((root / "manifests/audit.json").read_text())
    if audit.get("status") != "ok" or audit.get("manifests_written") is not True:
        raise ValueError("Manifest audit is not ready")
    selected = {}
    with (root / "manifests/train.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if row["split"] != "train":
                raise ValueError("Non-training row in train.csv")
            prefix = row["speaker_id"].split("_", 1)[0]
            if prefix not in PREFIXES or prefix in selected:
                continue
            path = Path(row["audio_path"]).resolve()
            path.relative_to(root / "audio")
            if not path.is_file() or "_Eng_" not in path.name:
                raise ValueError("Invalid training audio: " + str(path))
            selected[prefix] = {"speaker_id": row["speaker_id"], "audio_path": str(path)}
    missing = set(PREFIXES) - set(selected)
    if missing:
        raise ValueError("Missing training languages: " + ", ".join(sorted(missing)))
    return [selected[prefix] for prefix in PREFIXES]


def write_result(output, data):
    temporary = output / "result.json.tmp"
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(output / "result.json")


def run(args, clips):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Model execution requires SLURM_JOB_ID; use --plan on login nodes")
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    result = {"status": "running", "job_id": os.environ["SLURM_JOB_ID"],
              "model_id": MODEL_ID, "revision": REVISION, "clips": [],
              "max_seconds": args.max_seconds}
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA device in allocated job")
        device_name = torch.cuda.get_device_name(0)
        if "H200" not in device_name:
            raise RuntimeError("Expected H200, received " + device_name)
        provenance = json.loads((args.model_dir / "download.json").read_text())
        if provenance.get("revision") != REVISION or provenance.get("model_id") != MODEL_ID:
            raise ValueError("Model provenance mismatch")
        result.update(gpu=device_name, torch_version=torch.__version__,
                      cuda_version=torch.version.cuda)
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            str(args.model_dir), local_files_only=True)
        model = WavLMModel.from_pretrained(
            str(args.model_dir), local_files_only=True).to("cuda").eval()
        if model.config.hidden_size != 768:
            raise ValueError("Unexpected WavLM hidden size")
        embeddings = []
        for clip in clips:
            with sf.SoundFile(clip["audio_path"]) as audio:
                source_rate = audio.samplerate
                if audio.channels != 1 or source_rate not in (44100, 48000):
                    raise ValueError("Unexpected audio format: " + clip["audio_path"])
                waveform = audio.read(min(audio.frames, int(args.max_seconds * source_rate)),
                                      dtype="float32")
            if waveform.size == 0 or not np.isfinite(waveform).all():
                raise ValueError("Empty or non-finite audio")
            divisor = math.gcd(source_rate, TARGET_RATE)
            samples = resample_poly(waveform, TARGET_RATE // divisor,
                                    source_rate // divisor).astype(np.float32)
            expected = (len(waveform) * TARGET_RATE + source_rate - 1) // source_rate
            if len(samples) != expected or not np.isfinite(samples).all():
                raise ValueError("Resampling validation failed")
            inputs = extractor(samples, sampling_rate=TARGET_RATE,
                               return_tensors="pt", padding=False,
                               return_attention_mask=False)
            with torch.inference_mode():
                hidden = model(input_values=inputs.input_values.to("cuda")).last_hidden_state
                vector = hidden.mean(dim=1)[0].float().cpu().numpy()
            if vector.shape != (768,) or not np.isfinite(vector).all():
                raise ValueError("Invalid embedding")
            embeddings.append(vector)
            result["clips"].append(dict(clip, source_rate=source_rate,
                                        source_frames=len(waveform), target_rate=TARGET_RATE,
                                        target_frames=len(samples), feature_frames=hidden.shape[1]))
            print("Finished:", clip["speaker_id"], "768 finite features", flush=True)
        matrix = np.stack(embeddings)
        np.save(output / "embeddings.npy", matrix, allow_pickle=False)
        result.update(status="ok", embedding_shape=list(matrix.shape),
                      peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                      elapsed_seconds=time.monotonic() - started)
        write_result(output, result)
        print("NISP_GPU_SMOKE_OK", flush=True)
    except Exception as exc:
        result.update(status="error", error=str(exc),
                      elapsed_seconds=time.monotonic() - started)
        write_result(output, result)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nisp"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 10:
        parser.error("--max-seconds must be greater than 0 and at most 10")
    clips = select_clips(args.root)
    if args.plan:
        print(json.dumps({"revision": REVISION, "clips": clips,
                          "max_seconds": args.max_seconds}, indent=2))
        return
    if args.model_dir is None or args.output is None:
        parser.error("--model-dir and --output are required for execution")
    run(args, clips)


if __name__ == "__main__":
    main()
