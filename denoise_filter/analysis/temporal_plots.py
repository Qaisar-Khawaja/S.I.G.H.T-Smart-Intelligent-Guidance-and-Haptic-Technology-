"""
Part 10: report-ready plots for the temporal restoration experiment,
in the same style as analysis/plots.py (Dataset A).

  temporal_map_by_severity.png      mAP@0.5 per method, grouped by
                                     quality group -- the headline plot.
  temporal_recall_by_severity.png   Recall per method, grouped by
                                     quality group.
  temporal_latency.png              Mean total latency (preprocess +
                                     YOLO) per method, overall.
  temporal_rescued_vs_lost.png      Rescued vs lost ground-truth
                                     detections per temporal method
                                     (from results/temporal_frame_analysis.csv).

Usage:
    python -m analysis.temporal_plots
"""

import csv
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_CSV = "results/temporal_results.csv"
FRAME_ANALYSIS_CSV = "results/temporal_frame_analysis.csv"
OUTPUT_DIR = "results/plots"

METHOD_ORDER = ["raw", "wiener_denoise", "wiener_deconv", "clahe", "temporal_fixed", "temporal_quality_weighted"]
METHOD_COLORS = {
    "raw": "#888888",
    "wiener_denoise": "#ff7f0e",
    "wiener_deconv": "#d62728",
    "clahe": "#9467bd",
    "temporal_fixed": "#1f77b4",
    "temporal_quality_weighted": "#2ca02c",
}
SEVERITY_ORDER = ["clear", "moderate_blur", "severe_blur"]


def load_results():
    with open(RESULTS_CSV) as f:
        return list(csv.DictReader(f))


def plot_metric_by_severity(rows, metric, ylabel, out_name):
    fig, ax = plt.subplots(figsize=(9, 6))
    n_methods = len(METHOD_ORDER)
    x = np.arange(len(SEVERITY_ORDER))
    width = 0.8 / n_methods

    for i, method in enumerate(METHOD_ORDER):
        values = []
        for group in SEVERITY_ORDER:
            match = [r for r in rows if r["method"] == method and r["quality_group"] == group]
            values.append(float(match[0][metric]) if match else 0.0)
        ax.bar(x + i * width, values, width, label=method, color=METHOD_COLORS.get(method))

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(SEVERITY_ORDER)
    ax.set_xlabel("quality group")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by severity, per method (Dataset B, 84 real cane-camera frames)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, out_name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_latency(rows):
    fig, ax = plt.subplots(figsize=(7, 6))
    overall = {r["method"]: r for r in rows if r["quality_group"] == "overall"}

    methods = [m for m in METHOD_ORDER if m in overall]
    latencies = [float(overall[m]["total_latency_ms"]) for m in methods]
    colors = [METHOD_COLORS.get(m) for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(x, latencies, color=colors)
    for bar, lat in zip(bars, latencies):
        ax.annotate(f"{lat:.0f}ms", (bar.get_x() + bar.get_width() / 2, lat),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)

    ax.set_ylabel("mean total latency (ms) -- preprocess + YOLO")
    ax.set_title("Preprocessing + inference latency by method")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "temporal_latency.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_rescued_vs_lost():
    if not os.path.exists(FRAME_ANALYSIS_CSV):
        print(f"skipped rescued-vs-lost plot -- {FRAME_ANALYSIS_CSV} not found")
        return

    with open(FRAME_ANALYSIS_CSV) as f:
        rows = list(csv.DictReader(f))

    methods = sorted({r["method"] for r in rows})
    rescued = [sum(1 for r in rows if r["method"] == m and r["outcome"] == "rescued") for m in methods]
    lost = [sum(1 for r in rows if r["method"] == m and r["outcome"] == "lost") for m in methods]

    fig, ax = plt.subplots(figsize=(7, 6))
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, rescued, width, label="rescued (raw missed, temporal caught)", color="#2ca02c")
    ax.bar(x + width / 2, lost, width, label="lost (raw caught, temporal missed)", color="#d62728")

    for i, (r, l) in enumerate(zip(rescued, lost)):
        ax.annotate(str(r), (i - width / 2, r), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)
        ax.annotate(str(l), (i + width / 2, l), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("number of ground-truth object instances")
    ax.set_title("Rescued vs. lost detections, temporal restoration vs. raw")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "temporal_rescued_vs_lost.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = load_results()
    plot_metric_by_severity(rows, "map50", "mAP@0.5", "temporal_map_by_severity.png")
    plot_metric_by_severity(rows, "recall", "Recall", "temporal_recall_by_severity.png")
    plot_latency(rows)
    plot_rescued_vs_lost()


if __name__ == "__main__":
    main()
