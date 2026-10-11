#!/usr/bin/env python3
"""Check validation metrics independently and export the first-batch table/plot."""
import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nisp-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/nisp-xdg-cache")
Path(os.environ["XDG_CACHE_HOME"], "fontconfig").mkdir(parents=True, exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = (
    ("demographics", "2: Age + sex"),
    ("acoustic_demographics", "5: MFCC + age + sex"),
    ("ssl12_demographics", "6: WavLM layer 12 + age + sex"),
)


def render(evaluation, output):
    metrics = json.loads((evaluation / "validation_metrics.json").read_text())
    if metrics["scope"] != "validation_only" or metrics["target"] != "bmi":
        raise ValueError("Expected first-batch BMI validation only")
    with (evaluation / "validation_predictions.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows or any(row["split"] != "val" for row in rows):
        raise ValueError("Unexpected prediction split")
    if len({row["speaker_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate predictions")
    actual = np.array([float(row["bmi_actual"]) for row in rows])
    predictions = {}
    table = ["| Model | Alpha | MAE (kg/m²) | RMSE (kg/m²) | R² | n |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, label in MODELS:
        prediction = np.array([float(row[key]) for row in rows])
        if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
            raise ValueError("Nonfinite validation predictions")
        error = prediction - actual
        mae, rmse = float(np.abs(error).sum() / len(rows)), float(np.sqrt(np.dot(error, error) / len(rows)))
        recorded = metrics["models"][key]["validation"]
        if recorded["n"] != len(rows) or not np.isclose(mae, recorded["mae"]) or not np.isclose(rmse, recorded["rmse"]):
            raise ValueError("Independent error calculation differs")
        predictions[key] = prediction
        alpha = metrics["models"][key]["chosen_alpha"]
        table.append(f"| {label} | {alpha:g} | {mae:.3f} | {rmse:.3f} | {recorded['r2']:.3f} | {len(rows)} |")
    output.mkdir(parents=True, exist_ok=False)
    values = np.concatenate([actual] + list(predictions.values()))
    lower, upper = float(values.min() - 1), float(values.max() + 1)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained", sharex=True, sharey=True)
    for ax, (key, label) in zip(axes, MODELS):
        ax.scatter(actual, predictions[key], s=28, alpha=0.8, color="#2477a8", edgecolors="white", linewidths=0.4)
        ax.plot([lower, upper], [lower, upper], "--", color="#555555", linewidth=1)
        ax.set(xlim=(lower, upper), ylim=(lower, upper), xlabel="Actual BMI (kg/m²)", title=label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
    axes[0].set_ylabel("Predicted BMI (kg/m²)")
    fig.suptitle(f"NISP BMI: validation only (n={len(rows)})", fontsize=14)
    fig.savefig(output / "bmi-validation.png", dpi=200)
    fig.savefig(output / "bmi-validation.pdf")
    plt.close(fig)
    (output / "comparison.md").write_text(
        "# First-batch BMI validation\n\n" + "\n".join(table) +
        "\n\nModels were fitted on train and selected on validation. These are development scores, not final test results.\n",
        encoding="utf-8")
    (output / "metric-verification.json").write_text(json.dumps({
        "status": "ok", "n": len(rows), "mae_rmse_independently_recomputed": True,
        "test_metrics_computed": False,
    }, indent=2) + "\n")
    print(output / "bmi-validation.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.evaluation, args.output)
