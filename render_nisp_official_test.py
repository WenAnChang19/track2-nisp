#!/usr/bin/env python3
"""Render saved official test outputs; never fit models or choose settings."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODELS = ["mean", "demographics", "mfcc_demographics",
          "selected_ssl_demographics", "ssl12_demographics"]
LABELS = ["Development mean", "Age + sex OLS", "MFCC + age + sex",
          "Selected WavLM + age + sex", "WavLM layer 12 + age + sex"]
COLORS = ["#b8c5ce", "#667987", "#d49c42", "#17827c", "#6d77af"]
TARGETS = ["bmi", "height_cm"]
TITLES = {"bmi": "BMI", "height_cm": "Height"}
UNITS = {"bmi": "kg/m²", "height_cm": "cm"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, output, stem):
    fig.savefig(output / (stem + ".png"), dpi=180, bbox_inches="tight")
    fig.savefig(output / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def render(output):
    metrics = read_csv(output / "test_metrics.csv")
    comparisons = read_csv(output / "bootstrap_comparisons.csv")
    scored = read_csv(output / "test_scored_predictions.csv")
    groups = read_csv(output / "group_metrics.csv")
    assert len(metrics) == 10 and len(comparisons) == 8 and len(scored) == 270
    assert len(groups) == 70
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), constrained_layout=True)
    for target, ax in zip(TARGETS, axes):
        values = [float(next(r["mae"] for r in metrics
                             if r["target"] == target and r["model"] == model))
                  for model in MODELS]
        ax.barh(LABELS, values, color=COLORS)
        ax.invert_yaxis()
        ax.set_xlim(0, max(values) * 1.2)
        for i, value in enumerate(values):
            ax.text(value + max(values) * .015, i, f"{value:.3f}", va="center")
        ax.set_xlabel(f"Test MAE ({UNITS[target]}); lower is better")
        ax.set_title(TITLES[target] + " | 135 held-out speakers")
    save(fig, output, "test_mae_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    for target, ax in zip(TARGETS, axes):
        row = next(r for r in comparisons if r["target"] == target and r["kind"] == "primary")
        point = float(row["delta_mae_candidate_minus_reference"])
        for y, prefix, label, color in [(1, "ci95_", "Individual 95%", "#17827c"),
                                        (0, "family_ci97_5_", "Family-adjusted 97.5%", "#667987")]:
            lo, hi = float(row[prefix + "lower"]), float(row[prefix + "upper"])
            ax.hlines(y, lo, hi, lw=3, color=color)
            ax.plot([lo, hi], [y, y], "|", color=color, ms=12)
            ax.plot(point, y, "o", color=color)
            ax.text((lo + hi) / 2, y + .16, f"[{lo:.3f}, {hi:.3f}]", ha="center", fontsize=9)
        ax.axvline(0, color="#b9413a", linestyle="--", lw=1)
        ax.set_yticks([1, 0], ["Individual 95%", "Family-adjusted 97.5%"])
        ax.set_ylim(-.5, 1.6)
        ax.set_xlabel(f"MAE(selected SSL) − MAE(age + sex) ({UNITS[target]})")
        ax.set_title(TITLES[target] + f" | delta = {point:.3f}")
    fig.suptitle("Paired speaker bootstrap | 10,000 shared draws | negative favors selected SSL")
    save(fig, output, "primary_paired_intervals")

    selected_models = ["demographics", "mfcc_demographics", "selected_ssl_demographics"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for target, row_axes in zip(TARGETS, axes):
        rows = [r for r in scored if r["target"] == target]
        actual = np.asarray([float(r["actual"]) for r in rows])
        all_values = [actual] + [np.asarray([float(r[m + "_prediction"]) for r in rows])
                                  for m in selected_models]
        low, high = min(v.min() for v in all_values), max(v.max() for v in all_values)
        padding = (high - low) * .04
        limits = [low - padding, high + padding]
        for model, ax in zip(selected_models, row_axes):
            predicted = np.asarray([float(r[model + "_prediction"]) for r in rows])
            ax.scatter(actual, predicted, s=18, alpha=.65, color=COLORS[MODELS.index(model)])
            ax.plot(limits, limits, "--", color="#929ba2", lw=1)
            ax.set(xlim=limits, ylim=limits,
                   xlabel=f"Measured / derived {TITLES[target]} ({UNITS[target]})",
                   ylabel=f"Predicted {TITLES[target]} ({UNITS[target]})")
            mae = next(float(r["mae"]) for r in metrics if r["target"] == target and r["model"] == model)
            ax.set_title(LABELS[MODELS.index(model)] + f"\nMAE {mae:.3f}")
            ax.set_aspect("equal", adjustable="box")
    fig.suptitle("Official test predictions | identical 135 speakers in every panel")
    save(fig, output, "test_prediction_scatter")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    categories = [("sex", "Female"), ("sex", "Male"),
                  *[("language", lang) for lang in ["Hin", "Kan", "Mal", "Tam", "Tel"]]]
    for target, ax in zip(TARGETS, axes):
        rows = [next(r for r in groups if r["target"] == target
                     and r["model"] == "selected_ssl_demographics"
                     and (r["group_field"], r["group_value"]) == category)
                for category in categories]
        values = [float(r["delta_mae_vs_demographics"]) if int(r["n"]) else np.nan for r in rows]
        labels = [f"{r['group_value']} (n={r['n']})" for r in rows]
        ax.barh(labels, values, color=["#17827c" if v < 0 else "#c26f51" for v in values])
        ax.invert_yaxis()
        ax.axvline(0, color="#667987", lw=1)
        ax.set_xlabel(f"MAE(selected SSL) − MAE(age + sex) ({UNITS[target]})")
        ax.set_title(TITLES[target] + " | descriptive groups")
    fig.suptitle("Group differences describe this test set; no subgroup selection or significance claims")
    save(fig, output, "group_primary_deltas")

    evidence = {"purpose": "Saved-output visualization only; no fitting or selection",
                "input_file_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [output / "test_metrics.csv", output / "bootstrap_comparisons.csv",
                              output / "test_scored_predictions.csv", output / "group_metrics.csv"]},
                "render_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "png_and_pdf_stems": ["test_mae_comparison", "primary_paired_intervals",
                                      "test_prediction_scatter", "group_primary_deltas"]}
    (output / "plot_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": "ok", "figures": 4, "model_fit": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    render(args.results)
