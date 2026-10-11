#!/usr/bin/env python3
"""Download only the pinned WavLM files after explicit approval."""
import argparse
import hashlib
import json
from pathlib import Path

MODEL_ID = "microsoft/wavlm-base-plus"
REVISION = "4c66d4806a428f2e922ccfa1a962776e232d487b"
WEIGHT_SIZE = 377617425
WEIGHT_SHA256 = "3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51"
FILES = ("config.json", "preprocessor_config.json", "pytorch_model.bin")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/nisp"))
    parser.add_argument("--approved-download", action="store_true")
    args = parser.parse_args()
    if not args.approved_download:
        parser.error("Explicit approval required: about 378 MB of model files")
    from huggingface_hub import hf_hub_download

    root = args.root.resolve()
    directory = root / "models/wavlm-base-plus"
    cache = root / ".cache/huggingface/hub"
    directory.resolve().relative_to(root)
    cache.resolve().relative_to(root)
    directory.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        if (directory / name).is_symlink():
            raise ValueError("Refusing model file symlink")
        hf_hub_download(MODEL_ID, filename=name, revision=REVISION,
                        local_dir=str(directory), cache_dir=str(cache))
        print("Downloaded:", name, flush=True)
    weights = directory / "pytorch_model.bin"
    if weights.stat().st_size != WEIGHT_SIZE:
        raise ValueError("Weight file size mismatch")
    digest = hashlib.sha256()
    with weights.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("Weight SHA-256 mismatch")
    metadata = {"model_id": MODEL_ID, "revision": REVISION,
                "weight_bytes": WEIGHT_SIZE, "weight_sha256": digest.hexdigest(), "files": FILES}
    temporary = directory / "download.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(directory / "download.json")
    print("WAVLM_DOWNLOAD_OK", flush=True)


if __name__ == "__main__":
    main()
