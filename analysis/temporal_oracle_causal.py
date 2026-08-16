"""Split the completed temporal oracle into causal past/future directions.

This is a cheap post-processing step over temporal_oracle_instances.csv;
it does not rerun YOLO or alter the Phase-1 bidirectional results.
"""

import argparse
import csv
from pathlib import Path


INSTANCES_CSV = Path("results/temporal_oracle_instances.csv")
OUTPUT_CSV = Path("results/temporal_oracle_causal_summary.csv")
GROUPS = ("overall", "clear", "moderate_blur", "severe_blur")
WINDOWS = (1, 3, 5)


def _present(value):
    return value not in (None, "")


def summarize(rows):
    summaries = []
    for group in GROUPS:
        selected = rows if group == "overall" else [row for row in rows if row["quality_group"] == group]
        n_gt = len(selected)
        raw_count = sum(row["raw_detected"] == "True" for row in selected)
        summary = {
            "quality_group": group,
            "n_gt_instances": n_gt,
            "raw_detected": raw_count,
            "raw_recall": round(raw_count / n_gt, 4),
        }
        for window in WINDOWS:
            past_recovered = sum(
                row["raw_detected"] == "False"
                and _present(row["nearest_prev_detection_offset"])
                and abs(int(row["nearest_prev_detection_offset"])) <= window
                for row in selected
            )
            future_recovered = sum(
                row["raw_detected"] == "False"
                and _present(row["nearest_next_detection_offset"])
                and abs(int(row["nearest_next_detection_offset"])) <= window
                for row in selected
            )
            bidirectional_recovered = sum(row[f"recoverable_pm{window}"] == "True" for row in selected)
            summary[f"past{window}_recoverable_misses"] = past_recovered
            summary[f"past{window}_recall"] = round((raw_count + past_recovered) / n_gt, 4)
            summary[f"future{window}_recoverable_misses"] = future_recovered
            summary[f"future{window}_recall"] = round((raw_count + future_recovered) / n_gt, 4)
            summary[f"bidirectional_pm{window}_recoverable_misses"] = bidirectional_recovered
            summary[f"bidirectional_pm{window}_recall"] = round((raw_count + bidirectional_recovered) / n_gt, 4)
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-output", action="store_true")
    args = parser.parse_args()
    with INSTANCES_CSV.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 344:
        raise RuntimeError(f"Expected 344 oracle instances, found {len(rows)}")
    if OUTPUT_CSV.exists() and not args.force_output:
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_CSV}")

    summaries = summarize(rows)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print("group             raw   past1  past3  past5  future5  pm5")
    for row in summaries:
        print(
            f"{row['quality_group']:16s} {row['raw_recall']:.3f}  "
            f"{row['past1_recall']:.3f}   {row['past3_recall']:.3f}   {row['past5_recall']:.3f}   "
            f"{row['future5_recall']:.3f}    {row['bidirectional_pm5_recall']:.3f}"
        )
    print(f"Wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
