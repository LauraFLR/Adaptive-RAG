#!/usr/bin/env python3
"""Generate comprehensive evaluation charts from all_iterations.json."""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUT_DIR = os.path.join(os.path.dirname(__file__), "charts")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), "all_iterations.json")) as f:
    data = json.load(f)

# ── colour palette ──────────────────────────────────────────────────────
MODEL_COLORS = {"flan_t5_xl": "#4C72B0", "flan_t5_xxl": "#DD8452", "gpt": "#55A868"}
MODEL_LABELS = {"flan_t5_xl": "Flan-T5-XL", "flan_t5_xxl": "Flan-T5-XXL", "gpt": "GPT"}
ROUTE_COLORS = {"A": "#66c2a5", "B": "#fc8d62", "C": "#8da0cb"}
DATASET_ORDER = ["musique", "hotpotqa", "2wikimultihopqa", "nq", "trivia", "squad"]
DATASET_LABELS = {"musique": "MuSiQue", "hotpotqa": "HotpotQA",
                  "2wikimultihopqa": "2WikiMHQA", "nq": "NQ",
                  "trivia": "TriviaQA", "squad": "SQuAD"}

ITER_NAMES = {k: v["name"] for k, v in data.items()}
ITER_SHORT = {
    "IT1": "IT1\nBaseline",
    "IT2": "IT2\nUndersample",
    "IT3": "IT3\nWeighted CE",
    "IT4": "IT4\nFocal Loss",
    "IT5": "IT5\nAgreement",
    "IT7": "IT7\nUE Kappa",
}

sns.set_theme(style="whitegrid", font_scale=1.05)

def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# =========================================================================
# 1. Macro-Avg F1 & EM across iterations (grouped bar — one chart per metric)
# =========================================================================
def chart_macro_overview():
    iters_to_plot = ["IT1", "IT3", "IT4", "IT5", "IT7"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]

    for metric, label in [("macro_avg_f1", "Macro-Avg F1"), ("macro_avg_em", "Macro-Avg EM")]:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        x = np.arange(len(iters_to_plot))
        w = 0.22
        for j, m in enumerate(models):
            vals = []
            for it in iters_to_plot:
                mdata = data[it]["models"].get(m)
                vals.append(mdata["qa"][metric] if mdata else np.nan)
            bars = ax.bar(x + j * w, vals, w, label=MODEL_LABELS[m],
                          color=MODEL_COLORS[m], edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x + w)
        ax.set_xticklabels([ITER_SHORT[it] for it in iters_to_plot], fontsize=9)
        ax.set_ylabel(label)
        ax.set_title(f"{label} Across Iterations (All 3 Models)")
        ax.legend(loc="upper left", framealpha=0.9)
        ax.set_ylim(0.30, 0.58)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        save(fig, f"macro_{metric}.png")

chart_macro_overview()


# =========================================================================
# 2. Per-dataset EM heatmaps (one per model)
# =========================================================================
def chart_dataset_heatmaps():
    iters_to_plot = ["IT1", "IT3", "IT4", "IT5", "IT7"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    for m in models:
        matrix = []
        row_labels = []
        for it in iters_to_plot:
            mdata = data[it]["models"].get(m)
            if mdata is None:
                continue
            row = [mdata["qa"][ds]["em"] for ds in DATASET_ORDER]
            matrix.append(row)
            row_labels.append(f"{it}: {ITER_NAMES[it]}")
        matrix = np.array(matrix)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                    xticklabels=[DATASET_LABELS[d] for d in DATASET_ORDER],
                    yticklabels=row_labels, ax=ax, linewidths=0.5,
                    vmin=0.10, vmax=0.65)
        ax.set_title(f"Exact Match by Dataset — {MODEL_LABELS[m]}")
        save(fig, f"heatmap_em_{m}.png")

chart_dataset_heatmaps()


# =========================================================================
# 3. Per-dataset F1 heatmaps (one per model)
# =========================================================================
def chart_dataset_f1_heatmaps():
    iters_to_plot = ["IT1", "IT3", "IT4", "IT5", "IT7"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    for m in models:
        matrix = []
        row_labels = []
        for it in iters_to_plot:
            mdata = data[it]["models"].get(m)
            if mdata is None:
                continue
            row = [mdata["qa"][ds]["f1"] for ds in DATASET_ORDER]
            matrix.append(row)
            row_labels.append(f"{it}: {ITER_NAMES[it]}")
        matrix = np.array(matrix)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=[DATASET_LABELS[d] for d in DATASET_ORDER],
                    yticklabels=row_labels, ax=ax, linewidths=0.5,
                    vmin=0.20, vmax=0.78)
        ax.set_title(f"F1 Score by Dataset — {MODEL_LABELS[m]}")
        save(fig, f"heatmap_f1_{m}.png")

chart_dataset_f1_heatmaps()


# =========================================================================
# 4. Routing distribution stacked bar (one per model)
# =========================================================================
def chart_routing():
    iters_to_plot = ["IT1", "IT3", "IT4", "IT5", "IT7"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    for m in models:
        fig, ax = plt.subplots(figsize=(9, 5))
        labels, a_vals, b_vals, c_vals = [], [], [], []
        for it in iters_to_plot:
            mdata = data[it]["models"].get(m)
            if mdata is None:
                continue
            labels.append(ITER_SHORT[it])
            r = mdata["routing"]
            a_vals.append(r["A_pct"])
            b_vals.append(r["B_pct"])
            c_vals.append(r["C_pct"])
        x = np.arange(len(labels))
        ax.bar(x, a_vals, 0.5, label="A (no retrieval)", color=ROUTE_COLORS["A"])
        ax.bar(x, b_vals, 0.5, bottom=a_vals, label="B (single-step)", color=ROUTE_COLORS["B"])
        ax.bar(x, c_vals, 0.5, bottom=[a + b for a, b in zip(a_vals, b_vals)],
               label="C (multi-step)", color=ROUTE_COLORS["C"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Percentage of Questions")
        ax.set_title(f"Route Distribution — {MODEL_LABELS[m]}")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_ylim(0, 105)
        # Add total_steps annotation
        for i, it in enumerate([it for it in iters_to_plot if data[it]["models"].get(m)]):
            steps = data[it]["models"][m]["routing"]["total_steps"]
            ax.text(i, 102, f"{steps} steps", ha="center", va="bottom", fontsize=7.5, style="italic")
        save(fig, f"routing_{m}.png")

chart_routing()


# =========================================================================
# 5. Delta vs IT1 — radar / spider chart per model (IT5 vs IT1)
# =========================================================================
def chart_delta_radar():
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    for m in models:
        it5 = data["IT5"]["models"].get(m)
        if it5 is None or "delta_vs_IT1" not in it5:
            continue
        deltas = it5["delta_vs_IT1"]
        categories = [DATASET_LABELS[d] for d in DATASET_ORDER]
        values_f1 = [deltas[d]["delta_f1"] for d in DATASET_ORDER]
        values_em = [deltas[d]["delta_em"] for d in DATASET_ORDER]

        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        values_f1 += values_f1[:1]
        values_em += values_em[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values_f1, "o-", linewidth=2, label="ΔF1", color="#4C72B0")
        ax.fill(angles, values_f1, alpha=0.15, color="#4C72B0")
        ax.plot(angles, values_em, "s-", linewidth=2, label="ΔEM", color="#DD8452")
        ax.fill(angles, values_em, alpha=0.15, color="#DD8452")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"IT5 (Agreement Gate) vs IT1 (Baseline)\n{MODEL_LABELS[m]}", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        save(fig, f"radar_IT5_vs_IT1_{m}.png")

chart_delta_radar()


# =========================================================================
# 6. Classifier accuracy comparison (IT1 Clf1 + Clf2)
# =========================================================================
def chart_classifier_accuracy():
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Clf1 across iterations
    iters_clf1 = ["IT1", "IT3", "IT4"]
    ax = axes[0]
    x = np.arange(len(iters_clf1))
    w = 0.22
    for j, m in enumerate(models):
        vals = []
        for it in iters_clf1:
            mdata = data[it]["models"].get(m)
            if mdata and "clf1" in mdata:
                vals.append(mdata["clf1"]["overall_accuracy"])
            else:
                vals.append(np.nan)
        bars = ax.bar(x + j * w, vals, w, label=MODEL_LABELS[m], color=MODEL_COLORS[m],
                      edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x + w)
    ax.set_xticklabels([f"{it}\n{ITER_NAMES[it]}" for it in iters_clf1], fontsize=8)
    ax.set_ylabel("Overall Accuracy")
    ax.set_title("Clf1 (No-Ret vs Ret) Accuracy")
    ax.legend(fontsize=8)
    ax.set_ylim(0.55, 0.80)

    # Clf2 (IT1 only, the only one with Clf2 data)
    ax2 = axes[1]
    it1 = data["IT1"]
    clf2_vals = []
    clf2_labels_list = []
    clf2_colors = []
    for m in models:
        mdata = it1["models"].get(m)
        if mdata and "clf2" in mdata:
            clf2_vals.append(mdata["clf2"]["overall_accuracy"])
            clf2_labels_list.append(MODEL_LABELS[m])
            clf2_colors.append(MODEL_COLORS[m])
    bars2 = ax2.bar(clf2_labels_list, clf2_vals, color=clf2_colors, edgecolor="white", width=0.5)
    for bar, v in zip(bars2, clf2_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Overall Accuracy")
    ax2.set_title("Clf2 (Single vs Multi) Accuracy — IT1 Baseline")
    ax2.set_ylim(0.45, 0.80)

    fig.suptitle("Classifier Performance", fontsize=14, y=1.02)
    fig.tight_layout()
    save(fig, "classifier_accuracy.png")

chart_classifier_accuracy()


# =========================================================================
# 7. Efficiency: Total Steps vs Macro-Avg F1 scatter
# =========================================================================
def chart_efficiency():
    iters_to_plot = ["IT1", "IT3", "IT4", "IT5", "IT7"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    markers = {"flan_t5_xl": "o", "flan_t5_xxl": "s", "gpt": "D"}

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in models:
        steps_list, f1_list, labels_list = [], [], []
        for it in iters_to_plot:
            mdata = data[it]["models"].get(m)
            if mdata is None:
                continue
            steps_list.append(mdata["routing"]["total_steps"])
            f1_list.append(mdata["qa"]["macro_avg_f1"])
            labels_list.append(it)
        ax.scatter(steps_list, f1_list, label=MODEL_LABELS[m], color=MODEL_COLORS[m],
                   marker=markers[m], s=100, edgecolor="white", linewidth=0.5, zorder=3)
        for s, f, lab in zip(steps_list, f1_list, labels_list):
            ax.annotate(lab, (s, f), textcoords="offset points", xytext=(6, 6),
                        fontsize=7.5, color=MODEL_COLORS[m])
    ax.set_xlabel("Total Retrieval Steps")
    ax.set_ylabel("Macro-Avg F1")
    ax.set_title("Efficiency Frontier: Retrieval Cost vs QA Quality")
    ax.legend()
    ax.invert_xaxis()
    save(fig, "efficiency_frontier.png")

chart_efficiency()


# =========================================================================
# 8. IT6 Feature Probe — Logistic Regression Coefficients
# =========================================================================
def chart_kappa_probe():
    it6 = data["IT6"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    features = it6["features"]
    feature_labels = ["Token Len\n(norm)", "Entity\nDensity", "Hop\nDensity", "κ(q)"]
    slices = ["merged", "silver_only", "ib_only"]
    slice_labels = ["Merged", "Silver Only", "IB Only"]
    slice_colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for idx, m in enumerate(models):
        ax = axes[idx]
        x = np.arange(len(features))
        w = 0.25
        for si, sl in enumerate(slices):
            sdata = it6["models"][m][sl]
            coefs = [sdata["coefficients"][feat] for feat in features]
            ax.bar(x + si * w, coefs, w, label=slice_labels[si], color=slice_colors[si],
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x + w)
        ax.set_xticklabels(feature_labels, fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(MODEL_LABELS[m])
        if idx == 0:
            ax.set_ylabel("Logistic Regression Coefficient")
        if idx == 2:
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("IT6: κ(q) Feature Probe — Logistic Regression Coefficients", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "kappa_probe_coefficients.png")

chart_kappa_probe()


# =========================================================================
# 9. IT6 Feature Probe — ROC-AUC & Macro-F1 summary
# =========================================================================
def chart_kappa_probe_metrics():
    it6 = data["IT6"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    slices = ["merged", "silver_only", "ib_only"]
    slice_labels = ["Merged", "Silver Only", "IB Only"]
    slice_colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for mi, (metric, label) in enumerate([("roc_auc", "ROC-AUC"), ("macro_f1", "Macro-F1")]):
        ax = axes[mi]
        x = np.arange(len(models))
        w = 0.25
        for si, sl in enumerate(slices):
            vals = [it6["models"][m][sl][metric]["mean"] for m in models]
            errs = [it6["models"][m][sl][metric]["std"] for m in models]
            ax.bar(x + si * w, vals, w, yerr=errs, label=slice_labels[si],
                   color=slice_colors[si], edgecolor="white", linewidth=0.5,
                   capsize=3, error_kw={"linewidth": 1})
        ax.set_xticks(x + w)
        ax.set_xticklabels([MODEL_LABELS[m] for m in models])
        ax.set_ylabel(label)
        ax.set_title(f"IT6 Feature Probe — {label}")
        ax.set_ylim(0.45, 0.82)
        if mi == 1:
            ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "kappa_probe_metrics.png")

chart_kappa_probe_metrics()


# =========================================================================
# 10. GPT IT2 comparison (dedicated chart since IT2 is GPT-only)
# =========================================================================
def chart_gpt_it2():
    fig, ax = plt.subplots(figsize=(10, 5))
    it1_gpt = data["IT1"]["models"]["gpt"]["qa"]
    it2_gpt = data["IT2"]["models"]["gpt"]["qa"]
    x = np.arange(len(DATASET_ORDER))
    w = 0.3
    vals1 = [it1_gpt[d]["em"] for d in DATASET_ORDER]
    vals2 = [it2_gpt[d]["em"] for d in DATASET_ORDER]
    ax.bar(x - w/2, vals1, w, label="IT1: Baseline", color="#4C72B0", edgecolor="white")
    ax.bar(x + w/2, vals2, w, label="IT2: Undersampled Clf1", color="#DD8452", edgecolor="white")
    for i in range(len(DATASET_ORDER)):
        delta = vals2[i] - vals1[i]
        color = "#2ca02c" if delta >= 0 else "#d62728"
        ax.text(x[i] + w/2, vals2[i] + 0.008, f"{delta:+.3f}", ha="center",
                va="bottom", fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASET_ORDER])
    ax.set_ylabel("Exact Match")
    ax.set_title("GPT: IT1 Baseline vs IT2 Undersampled Clf1")
    ax.legend()
    ax.set_ylim(0.10, 0.72)
    save(fig, "gpt_it2_comparison.png")

chart_gpt_it2()


# =========================================================================
# 11. Summary table — best iteration per model
# =========================================================================
def chart_summary_table():
    iters_all = ["IT1", "IT3", "IT4", "IT5", "IT7"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis("off")

    header = ["Model", "Best Iter (F1)", "Best F1", "Best Iter (EM)", "Best EM",
              "Best Steps", "IT5 F1", "IT5 EM"]
    rows = []
    for m in models:
        best_f1_it, best_f1_val = "", 0
        best_em_it, best_em_val = "", 0
        best_steps = ""
        it5_f1, it5_em = "", ""
        for it in iters_all:
            mdata = data[it]["models"].get(m)
            if mdata is None:
                continue
            f1 = mdata["qa"]["macro_avg_f1"]
            em = mdata["qa"]["macro_avg_em"]
            if f1 > best_f1_val:
                best_f1_val = f1
                best_f1_it = it
            if em > best_em_val:
                best_em_val = em
                best_em_it = it
        it5_data = data["IT5"]["models"].get(m)
        if it5_data:
            it5_f1 = f"{it5_data['qa']['macro_avg_f1']:.4f}"
            it5_em = f"{it5_data['qa']['macro_avg_em']:.4f}"
            best_steps = str(it5_data["routing"]["total_steps"])
        rows.append([MODEL_LABELS[m], best_f1_it, f"{best_f1_val:.4f}",
                     best_em_it, f"{best_em_val:.4f}", best_steps, it5_f1, it5_em])

    table = ax.table(cellText=rows, colLabels=header, loc="center",
                     cellLoc="center", colColours=["#d4e6f1"] * len(header))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    # Bold best row
    for i, row in enumerate(rows):
        for j in range(len(header)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#bbb")
    ax.set_title("Summary: Best Iteration per Model", fontsize=13, pad=20)
    save(fig, "summary_table.png")

chart_summary_table()


# =========================================================================
# 12. Combined Macro F1 line chart (all iterations, all models)
# =========================================================================
def chart_macro_line():
    iters_order = ["IT1", "IT3", "IT4", "IT5", "IT7"]
    models = ["flan_t5_xl", "flan_t5_xxl", "gpt"]
    markers = {"flan_t5_xl": "o", "flan_t5_xxl": "s", "gpt": "D"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in models:
        vals = []
        x_pos = []
        for i, it in enumerate(iters_order):
            mdata = data[it]["models"].get(m)
            if mdata:
                vals.append(mdata["qa"]["macro_avg_f1"])
                x_pos.append(i)
        ax.plot(x_pos, vals, marker=markers[m], label=MODEL_LABELS[m],
                color=MODEL_COLORS[m], linewidth=2, markersize=8)
        for xp, v in zip(x_pos, vals):
            ax.annotate(f"{v:.3f}", (xp, v), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=8, color=MODEL_COLORS[m])
    ax.set_xticks(range(len(iters_order)))
    ax.set_xticklabels([ITER_SHORT[it] for it in iters_order], fontsize=9)
    ax.set_ylabel("Macro-Avg F1")
    ax.set_title("Macro-Avg F1 Progression Across Iterations")
    ax.legend()
    ax.set_ylim(0.40, 0.56)
    save(fig, "macro_f1_line.png")

chart_macro_line()


print("\nAll charts generated in", OUT_DIR)
