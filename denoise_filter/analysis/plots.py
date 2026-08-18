"""
Turns results/results_dataset_a.csv into the report-ready figures:

  severity_curves.png   -- mAP@0.5 vs degradation severity, one line per
                            filter, one subplot per degradation type.
                            This is the core "does the best filter
                            depend on degradation severity" plot.
  psnr_vs_map.png        -- scatter of PSNR against mAP@0.5 across every
                            condition, to visualize when the two
                            metrics disagree (a filter can win on image
                            quality but lose on detection, or vice versa).
  latency_vs_accuracy.png -- mAP@0.5 vs total latency, averaged per
                            filter across all conditions -- the
                            accuracy/speed trade-off.

Usage:
    python -m analysis.plots
"""

import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_CSV = "results/results_dataset_a.csv"
OUTPUT_DIR = "results/plots"

DEGRADATIONS = ["noise", "blur", "low_light", "glare"]

# degrade.py names noise's top tier "strong" while blur/low_light/glare
# use "severe" -- keep each degradation's own severity order rather
# than assuming one shared list, or the odd one out silently drops its
# third data point.
SEVERITY_ORDER = {
    "noise": ["mild", "medium", "strong"],
    "blur": ["mild", "medium", "severe"],
    "low_light": ["mild", "medium", "severe"],
    "glare": ["mild", "medium", "severe"],
}
FILTER_COLORS = {
    "raw": "#888888",
    "gaussian": "#1f77b4",
    "bilateral": "#2ca02c",
    "wiener_denoise": "#ff7f0e",
    "wiener_deconv": "#d62728",
    "clahe": "#9467bd",
}


def load_rows():
    with open(INPUT_CSV) as f:
        return list(csv.DictReader(f))


def plot_severity_curves(rows):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for ax, degradation in zip(axes, DEGRADATIONS):
        by_filter = defaultdict(dict)
        for row in rows:
            if row["degradation"] != degradation:
                continue
            by_filter[row["filter"]][row["severity"]] = float(row["map50"])

        order = SEVERITY_ORDER[degradation]
        for filter_name, severities in by_filter.items():
            y = [severities.get(s, None) for s in order]
            ax.plot(
                order, y, marker="o", label=filter_name,
                color=FILTER_COLORS.get(filter_name),
            )

        ax.set_title(degradation)
        ax.set_xlabel("severity")
        ax.set_ylabel("mAP@0.5")
        ax.set_ylim(0, 0.5)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("YOLO mAP@0.5 vs degradation severity, by filter", y=1.06)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "severity_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_psnr_vs_map(rows):
    fig, ax = plt.subplots(figsize=(8, 6))

    for row in rows:
        psnr = row["psnr"]
        if psnr in ("inf", "Infinity"):
            continue  # undegraded "none" + raw case, not comparable
        ax.scatter(
            float(psnr), float(row["map50"]),
            color=FILTER_COLORS.get(row["filter"], "black"),
            alpha=0.7, s=40,
        )

    for filter_name, color in FILTER_COLORS.items():
        ax.scatter([], [], color=color, label=filter_name)  # legend proxy

    ax.set_xlabel("PSNR (dB) -- image reconstruction quality")
    ax.set_ylabel("mAP@0.5 -- YOLO detection performance")
    ax.set_title("Image quality vs. detection performance, across all conditions")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "psnr_vs_map.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def plot_latency_vs_accuracy(rows):
    by_filter_map = defaultdict(list)
    by_filter_latency = defaultdict(list)
    for row in rows:
        by_filter_map[row["filter"]].append(float(row["map50"]))
        by_filter_latency[row["filter"]].append(float(row["total_ms"]))

    fig, ax = plt.subplots(figsize=(7, 6))
    for filter_name in by_filter_map:
        mean_map = sum(by_filter_map[filter_name]) / len(by_filter_map[filter_name])
        mean_latency = sum(by_filter_latency[filter_name]) / len(by_filter_latency[filter_name])
        ax.scatter(mean_latency, mean_map, s=100, color=FILTER_COLORS.get(filter_name))
        ax.annotate(filter_name, (mean_latency, mean_map), xytext=(6, 4), textcoords="offset points")

    ax.set_xlabel("mean total latency (ms) -- preprocess + YOLO")
    ax.set_ylabel("mean mAP@0.5 across all conditions")
    ax.set_title("Accuracy vs. computational cost")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "latency_vs_accuracy.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = load_rows()
    plot_severity_curves(rows)
    plot_psnr_vs_map(rows)
    plot_latency_vs_accuracy(rows)


if __name__ == "__main__":
    main()
