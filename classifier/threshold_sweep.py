"""
Threshold sweep for Clf1 (A vs R).

Loads a trained T5-large model, runs inference on the validation set to get
per-class probabilities, then sweeps the A-prediction threshold and reports
precision / recall / F1 / accuracy / macro-F1 at each operating point.

Usage:
    python classifier/threshold_sweep.py \
        --model_path classifier/outputs/.../epoch/20/.../  \
        --valid_file classifier/data/.../no_retrieval_vs_retrieval/valid.json \
        --labels A R \
        --output_dir classifier/outputs/.../threshold_sweep/
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset
from torch.utils.data import DataLoader


def get_probs(model, tokenizer, dataloader, labels, device):
    """Run inference and return (N, num_labels) probability matrix."""
    label_token_ids = [tokenizer(label).input_ids[0] for label in labels]
    all_probs = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                return_dict_in_generate=True,
                output_scores=True,
                max_length=30,
            )
            scores = out.scores[0]  # (batch, vocab)
            logits_for_labels = torch.stack(
                [scores[:, tid] for tid in label_token_ids], dim=1
            )  # (batch, num_labels)
            probs = torch.nn.functional.softmax(logits_for_labels, dim=1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def sweep(probs, gold, labels, thresholds):
    """Sweep threshold for the first label (A) and compute metrics."""
    a_idx = 0  # index of A in labels
    results = []
    for t in thresholds:
        preds = []
        for p in probs:
            if p[a_idx] >= t:
                preds.append(labels[a_idx])
            else:
                preds.append(labels[np.argmax(p[1:]) + 1])
        preds = np.array(preds)
        gold_arr = np.array(gold)

        metrics = {"threshold": round(t, 4)}
        macro_f1_parts = []
        for i, label in enumerate(labels):
            tp = int(np.sum((preds == label) & (gold_arr == label)))
            fp = int(np.sum((preds == label) & (gold_arr != label)))
            fn = int(np.sum((preds != label) & (gold_arr == label)))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            metrics[f"{label}_precision"] = round(prec, 4)
            metrics[f"{label}_recall"] = round(rec, 4)
            metrics[f"{label}_f1"] = round(f1, 4)
            metrics[f"{label}_pred_count"] = int(np.sum(preds == label))
            macro_f1_parts.append(f1)

        metrics["overall_acc"] = round(float(np.mean(preds == gold_arr)), 4)
        metrics["macro_f1"] = round(float(np.mean(macro_f1_parts)), 4)
        results.append(metrics)
    return results


def plot_sweep(results, labels, output_path):
    thresholds = [r["threshold"] for r in results]
    a_label = labels[0]
    r_label = labels[1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, [r[f"{a_label}_recall"] for r in results], label=f"{a_label} recall", linewidth=2)
    ax.plot(thresholds, [r[f"{r_label}_recall"] for r in results], label=f"{r_label} recall", linewidth=2)
    ax.plot(thresholds, [r["overall_acc"] for r in results], label="Overall accuracy", linewidth=2, linestyle="--")
    ax.plot(thresholds, [r["macro_f1"] for r in results], label="Macro F1", linewidth=2, linestyle=":")

    # Mark default (0.5) line
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Default threshold (0.5)")

    ax.set_xlabel(f"Threshold for predicting {a_label}", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Clf1 Threshold Sweep: {a_label} vs {r_label}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def find_special_thresholds(results, labels, default_acc):
    a_label = labels[0]
    r_label = labels[1]

    # 1. Max A-recall within 3pp of default accuracy
    best_a_recall = None
    for r in results:
        if r["overall_acc"] >= default_acc - 0.03:
            if best_a_recall is None or r[f"{a_label}_recall"] > best_a_recall[f"{a_label}_recall"]:
                best_a_recall = r

    # 2. Max macro-F1
    best_macro = max(results, key=lambda r: r["macro_f1"])

    # 3. Balanced operating point (minimize |A_recall - R_recall|)
    balanced = min(results, key=lambda r: abs(r[f"{a_label}_recall"] - r[f"{r_label}_recall"]))

    return best_a_recall, best_macro, balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--labels", type=str, nargs="+", default=["A", "R"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--default_acc", type=float, default=None,
                        help="Default accuracy at argmax threshold. If not given, computed from 0.5 threshold.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)

    print(f"Loading validation data from {args.valid_file}")
    raw = load_dataset("json", data_files={"validation": args.valid_file})

    def tokenize(examples):
        inputs = tokenizer(
            [q.strip() for q in examples["question"]],
            max_length=384,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            text_target=examples["answer"],
            max_length=30,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    ds = raw["validation"].map(tokenize, batched=True, remove_columns=raw["validation"].column_names)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    gold = raw["validation"]["answer"]

    print("Running inference to get probabilities...")
    probs = get_probs(model, tokenizer, dataloader, args.labels, device)
    print(f"Got probabilities for {probs.shape[0]} samples, shape {probs.shape}")

    # Save raw probabilities
    np.save(os.path.join(args.output_dir, "probs.npy"), probs)

    # Sweep thresholds
    thresholds = np.arange(0.10, 0.91, 0.05).tolist()
    # Also add finer grain around argmax region
    thresholds = sorted(set([round(t, 4) for t in thresholds]))

    print(f"Sweeping {len(thresholds)} thresholds...")
    results = sweep(probs, gold, args.labels, thresholds)

    # Default accuracy (at 0.5, equivalent to argmax for 2 classes)
    default_result = [r for r in results if abs(r["threshold"] - 0.5) < 0.01]
    default_acc = default_result[0]["overall_acc"] if default_result else (args.default_acc or 0.7319)
    if args.default_acc:
        default_acc = args.default_acc

    # Print table
    a_label, r_label = args.labels[0], args.labels[1]
    header = f"{'Thresh':>7} | {a_label+'-P':>6} {a_label+'-R':>6} {a_label+'-F1':>6} | {r_label+'-P':>6} {r_label+'-R':>6} {r_label+'-F1':>6} | {'Acc':>6} {'MacF1':>6} | {a_label+'#':>5} {r_label+'#':>5}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r['threshold']:7.2f} | {r[f'{a_label}_precision']:6.3f} {r[f'{a_label}_recall']:6.3f} {r[f'{a_label}_f1']:6.3f} | {r[f'{r_label}_precision']:6.3f} {r[f'{r_label}_recall']:6.3f} {r[f'{r_label}_f1']:6.3f} | {r['overall_acc']:6.3f} {r['macro_f1']:6.3f} | {r[f'{a_label}_pred_count']:5d} {r[f'{r_label}_pred_count']:5d}")
    print("=" * len(header))

    # Find special thresholds
    best_a_recall, best_macro, balanced = find_special_thresholds(results, args.labels, default_acc)

    print(f"\n--- Special operating points (default acc = {default_acc:.4f}) ---")
    if best_a_recall:
        print(f"\n1. Max {a_label}-recall within 3pp of default acc:")
        print(f"   Threshold = {best_a_recall['threshold']:.2f}, {a_label}-recall = {best_a_recall[f'{a_label}_recall']:.3f}, "
              f"Acc = {best_a_recall['overall_acc']:.3f}, Macro-F1 = {best_a_recall['macro_f1']:.3f}")
    else:
        print(f"\n1. No threshold found within 3pp of default acc")

    print(f"\n2. Max macro-F1:")
    print(f"   Threshold = {best_macro['threshold']:.2f}, Macro-F1 = {best_macro['macro_f1']:.3f}, "
          f"{a_label}-recall = {best_macro[f'{a_label}_recall']:.3f}, Acc = {best_macro['overall_acc']:.3f}")

    print(f"\n3. Balanced operating point ({a_label}-recall ≈ {r_label}-recall):")
    print(f"   Threshold = {balanced['threshold']:.2f}, {a_label}-recall = {balanced[f'{a_label}_recall']:.3f}, "
          f"{r_label}-recall = {balanced[f'{r_label}_recall']:.3f}, Acc = {balanced['overall_acc']:.3f}")

    # Plot
    plot_path = os.path.join(args.output_dir, "threshold_sweep.png")
    plot_sweep(results, args.labels, plot_path)

    # Save results
    with open(os.path.join(args.output_dir, "threshold_sweep_results.json"), "w") as f:
        json.dump({
            "default_acc": default_acc,
            "results": results,
            "best_a_recall_within_3pp": best_a_recall,
            "best_macro_f1": best_macro,
            "balanced_point": balanced,
        }, f, indent=2)
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
