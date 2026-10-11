#!/usr/bin/env python3
"""Record only GPU-related diagnostics inside an allocated Slurm step."""
import json
import os
import subprocess

if not os.environ.get("SLURM_JOB_ID"):
    raise SystemExit("GPU diagnostics requires Slurm")
import torch

record = {
    "job_id": os.environ["SLURM_JOB_ID"],
    "hostname": os.uname().nodename,
    "torch_version": torch.__version__, "torch_cuda_build": torch.version.cuda,
    "environment": {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_STEP_GRES",
        "LD_LIBRARY_PATH")},
}
try:
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"
    ], capture_output=True, text=True, timeout=15)
    record.update(nvidia_smi_exit_code=result.returncode,
                  nvidia_smi_stdout=result.stdout, nvidia_smi_stderr=result.stderr)
except (OSError, subprocess.TimeoutExpired) as exc:
    record["nvidia_smi_error"] = str(exc)
record.update(cuda_available=torch.cuda.is_available(), device_count=torch.cuda.device_count())
try:
    torch.cuda.init()
    record["device_name"] = torch.cuda.get_device_name(0)
except Exception as exc:
    record["cuda_initialization_error"] = str(exc)
print(json.dumps(record, indent=2), flush=True)
if not record["cuda_available"] or "H200" not in record.get("device_name", ""):
    raise SystemExit(2)
