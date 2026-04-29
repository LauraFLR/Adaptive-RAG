#!/usr/bin/env python3
"""Print per-dataset, per-class label distributions for all Iteration 1 data stages
and write the results to a JSON file."""

import json
import os
from collections import Counter

BASE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "classifier", "data", "musique_hotpot_wiki2_nq_tqa_sqd",
)
OUTPUT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "label_distribution_per_dataset.json",
)

MODELS = ["flan_t5_xl", "flan_t5_xxl", "gpt"]

DATASETS_ORDER = [
    "musique", "hotpotqa", "2wikimultihopqa", "nq", "trivia", "squad",
]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def count_labels(data, label_field="answer", dataset_field="dataset_name"):
    """Return {dataset: Counter(label -> count)}."""
    grouped = {}
    for row in data:
        ds = row[dataset_field]
        lbl = row[label_field]
        grouped.setdefault(ds, Counter())[lbl] += 1
    return grouped


def grouped_to_dict(grouped, labels):
    """Convert grouped counters to a serialisable dict with totals."""
    result = {}
    totals = Counter()
    grand_total = 0
    for ds in DATASETS_ORDER:
        if ds not in grouped:
            continue
        row = {l: grouped[ds].get(l, 0) for l in labels}
        row["total"] = sum(row.values())
        grand_total += row["total"]
        for l in labels:
            totals[l] += row[l]
        result[ds] = row
    total_row = {l: totals[l] for l in labels}
    total_row["total"] = grand_total
    result["TOTAL"] = total_row
    return result


def print_table(grouped, labels, title):
    """Print a formatted table: dataset × label counts with totals."""
    print(f"\n  {title}")
    ds_width = max(len(d) for d in DATASETS_ORDER + ["TOTAL"])
    col_w = max(7, max(len(l) for l in labels) + 2)

    header = f"    {'dataset':<{ds_width}}"
    for l in labels:
        header += f"  {l:>{col_w}}"
    header += f"  {'Total':>{col_w}}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    totals = Counter()
    grand_total = 0
    for ds in DATASETS_ORDER:
        if ds not in grouped:
            continue
        row_total = sum(grouped[ds].values())
        grand_total += row_total
        line = f"    {ds:<{ds_width}}"
        for l in labels:
            c = grouped[ds].get(l, 0)
            totals[l] += c
            line += f"  {c:>{col_w}}"
        line += f"  {row_total:>{col_w}}"
        print(line)

    print("    " + "-" * (len(header) - 4))
    line = f"    {'TOTAL':<{ds_width}}"
    for l in labels:
        line += f"  {totals[l]:>{col_w}}"
    line += f"  {grand_total:>{col_w}}"
    print(line)


def process_file(path, labels, title, json_out):
    """Load a single file, print its table, store in json_out. Returns data or None."""
    data = load_json(path)
    if data is None:
        print(f"\n  {title}")
        print(f"    WARNING: file not found — {path}")
        json_out[title] = {"warning": f"file not found — {path}"}
        return None
    grouped = count_labels(data)
    rel = os.path.relpath(path, BASE)
    full_title = f"{title}  (n={len(data)})  [{rel}]"
    print_table(grouped, labels, full_title)
    json_out[title] = {
        "file": rel,
        "n": len(data),
        "labels": labels,
        "per_dataset": grouped_to_dict(grouped, labels),
    }
    return data


def main():
    all_results = {}

    for model in MODELS:
        print("\n" + "=" * 80)
        print(f"MODEL: {model}")
        print("=" * 80)

        m = os.path.join(BASE, model)
        model_results = {}

        # ── 1. Silver 3-class ────────────────────────────────────────────
        print("\n── 1. Silver 3-class (A / B / C) ──")
        process_file(
            os.path.join(m, "silver", "train.json"),
            ["A", "B", "C"], "silver_3class_train", model_results,
        )
        process_file(
            os.path.join(m, "silver", "valid.json"),
            ["A", "B", "C"], "silver_3class_valid", model_results,
        )

        # ── 2. Clf1: A vs R ─────────────────────────────────────────────
        print("\n── 2. Clf1 — No-retrieval (A) vs Retrieval (R) ──")
        process_file(
            os.path.join(m, "silver", "no_retrieval_vs_retrieval", "train.json"),
            ["A", "R"], "clf1_A_vs_R_train", model_results,
        )
        process_file(
            os.path.join(m, "silver", "no_retrieval_vs_retrieval", "valid.json"),
            ["A", "R"], "clf1_A_vs_R_valid", model_results,
        )

        # ── 3. Clf2: B vs C ─────────────────────────────────────────────
        print("\n── 3. Clf2 — Single-step (B) vs Multi-step (C) ──")

        # 3a. IB-only (heuristic) — derived by set difference
        merged_path = os.path.join(m, "binary_silver_single_vs_multi", "train.json")
        silver_bc_path = os.path.join(m, "silver", "single_vs_multi", "train.json")
        merged_data = load_json(merged_path)
        silver_bc_data = load_json(silver_bc_path)

        if merged_data is not None and silver_bc_data is not None:
            silver_bc_ids = {row["id"] for row in silver_bc_data}
            ib_only = [row for row in merged_data if row["id"] not in silver_bc_ids]
            grouped = count_labels(ib_only)
            labels_bc = ["B", "C"]
            print_table(
                grouped, labels_bc,
                f"IB-only heuristic (derived: merged − silver B/C)  (n={len(ib_only)})  "
                f"[set difference of {os.path.relpath(merged_path, BASE)} − {os.path.relpath(silver_bc_path, BASE)}]",
            )
            model_results["clf2_ib_only_train"] = {
                "file": f"set difference of {os.path.relpath(merged_path, BASE)} − {os.path.relpath(silver_bc_path, BASE)}",
                "n": len(ib_only),
                "labels": labels_bc,
                "per_dataset": grouped_to_dict(grouped, labels_bc),
            }
        else:
            print("\n  IB-only heuristic")
            warnings = []
            if merged_data is None:
                print(f"    WARNING: file not found — {merged_path}")
                warnings.append(f"file not found — {merged_path}")
            if silver_bc_data is None:
                print(f"    WARNING: file not found — {silver_bc_path}")
                warnings.append(f"file not found — {silver_bc_path}")
            model_results["clf2_ib_only_train"] = {"warning": "; ".join(warnings)}

        # 3b. Silver B/C subset
        process_file(
            silver_bc_path,
            ["B", "C"], "clf2_silver_bc_train", model_results,
        )

        # 3c. Merged train
        process_file(
            merged_path,
            ["B", "C"], "clf2_merged_train", model_results,
        )

        # 3d. Merged valid — actual path is silver/single_vs_multi/valid.json
        process_file(
            os.path.join(m, "silver", "single_vs_multi", "valid.json"),
            ["B", "C"], "clf2_valid", model_results,
        )

        all_results[model] = model_results

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nJSON results written to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
