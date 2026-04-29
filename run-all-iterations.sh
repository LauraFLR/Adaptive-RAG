#!/usr/bin/env bash
# run-all-iterations.sh — Run all 7 Adaptive-RAG cascade iterations.
#
# Iterations:
#   1: Normal cascade (standard Clf1 + standard Clf2)
#   2: Cascade w/ Clf1 undersampling
#   3: Cascade w/ Clf1 weighted CE
#   4: Cascade w/ Clf1 focal loss
#   5: Cascade w/ Clf1 UE agreement gate
#   6: Cascade w/ Clf2 κ(q) feature probing
#   7: Cascade w/ UE kappa (fully training-free)
#
# Usage:
#   cd /root/laura/Adaptive-RAG
#   bash run-all-iterations.sh           # run everything
#   bash run-all-iterations.sh 5 6 7     # run only IT5, IT6, IT7

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
CLF_DIR="${REPO_ROOT}/classifier"
DATASET=musique_hotpot_wiki2_nq_tqa_sqd
MODELS=(flan_t5_xl flan_t5_xxl gpt)

# Save args BEFORE source can clobber $@
SAVED_ARGS=("$@")

source /root/laura/adaptiverag/bin/activate
cd "${REPO_ROOT}"

if [[ ${#SAVED_ARGS[@]} -gt 0 ]]; then
    ITERATIONS=("${SAVED_ARGS[@]}")
else
    ITERATIONS=(1 2 3 4 5 6 7)
fi

SKIP_PHASE0=${SKIP_PHASE0:-false}
SKIP_TRAINING=${SKIP_TRAINING:-false}

should_run() {
    for i in "${ITERATIONS[@]}"; do [[ "$i" == "$1" ]] && return 0; done
    return 1
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

script_tag() {
    case "$1" in
        flan_t5_xl)  echo "xl" ;;
        flan_t5_xxl) echo "xxl" ;;
        gpt)         echo "gpt" ;;
    esac
}

# find_best_epoch BASE_EPOCH_DIR VALID_FILE [RUN_TAG]
#
# Searches all epoch runs under BASE_EPOCH_DIR, compares validation
# predictions to ground truth, and prints the predict/dict_id_pred_results.json
# path for the epoch with the highest validation accuracy.
#
# RUN_TAG (optional): if given, only considers runs whose path contains
# this tag (e.g. "silver_only", "feat").  If omitted, excludes known
# non-standard tags so only timestamp-based runs are compared.
find_best_epoch() {
    python - "$@" <<'PYEOF'
import sys, os, json, glob

base_path = sys.argv[1]
valid_file = sys.argv[2]
run_tag = sys.argv[3] if len(sys.argv) > 3 else ""

with open(valid_file) as f:
    data = json.load(f)
gold = {item["id"]: item["answer"] for item in data}

KNOWN_TAGS = {"silver_only", "feat"}

# First pass: find the latest timestamp across all epoch directories
latest_ts = ""
for vf in glob.glob(os.path.join(base_path, "**/valid/dict_id_pred_results.json"), recursive=True):
    run_dir = os.path.dirname(os.path.dirname(vf))
    rel = os.path.relpath(run_dir, base_path)
    parts = rel.split(os.sep)
    if run_tag:
        if run_tag not in parts:
            continue
    else:
        if any(p in KNOWN_TAGS for p in parts):
            continue
    if len(parts) >= 3:
        ts = parts[1] + "/" + parts[2]
        if ts > latest_ts:
            latest_ts = ts

# Second pass: find best epoch within the latest timestamp only
best_acc = -1
best_path = ""
best_label = ""

for vf in glob.glob(os.path.join(base_path, "**/valid/dict_id_pred_results.json"), recursive=True):
    run_dir = os.path.dirname(os.path.dirname(vf))
    rel = os.path.relpath(run_dir, base_path)
    parts = rel.split(os.sep)

    if run_tag:
        if run_tag not in parts:
            continue
    else:
        if any(p in KNOWN_TAGS for p in parts):
            continue

    # Only consider runs from the latest timestamp
    if len(parts) >= 3:
        ts = parts[1] + "/" + parts[2]
        if ts != latest_ts:
            continue

    pf = os.path.join(run_dir, "predict", "dict_id_pred_results.json")
    if not os.path.exists(pf):
        continue

    with open(vf) as f:
        preds = json.load(f)

    correct = sum(1 for qid, info in preds.items()
                  if gold.get(qid) == info.get("prediction"))
    acc = correct / len(preds) if preds else 0

    if acc > best_acc:
        best_acc = acc
        best_path = pf
        best_label = rel

if best_path:
    print(best_path)
    print(f"  Best epoch: {best_label}  (val acc {best_acc:.4f})", file=sys.stderr)
else:
    print(f"ERROR: no valid predictions found under {base_path}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

# route_split ITER_TAG MODEL CLF1_PRED CLF2_PRED
route_split() {
    local tag="$1" model="$2" clf1="$3" clf2="$4"
    local out="predictions/classifier/t5-large/${model}/${tag}"
    echo "  [${tag}/${model}] Routing (Clf1 + Clf2)..."
    python classifier/postprocess/predict_complexity_split_classifiers.py "${model}" \
        --no_ret_vs_ret_file "${clf1}" \
        --single_vs_multi_file "${clf2}" \
        --output_path "${out}"
    echo "  [${tag}/${model}] Evaluating..."
    python evaluate_final_acc.py --pred_path "${out}"
}

# route_agreement ITER_TAG MODEL CLF2_PRED
route_agreement() {
    local tag="$1" model="$2" clf2="$3"
    local out="predictions/classifier/t5-large/${model}/${tag}"
    echo "  [${tag}/${model}] Routing (agreement gate + Clf2)..."
    python classifier/postprocess/predict_complexity_agreement.py "${model}" \
        --clf2_pred_file "${clf2}" \
        --predict_file "classifier/data/${DATASET}/predict.json" \
        --output_path "${out}"
    echo "  [${tag}/${model}] Evaluating..."
    python evaluate_final_acc.py --pred_path "${out}"
}

# route_kappa ITER_TAG MODEL
route_kappa() {
    local tag="$1" model="$2"
    local out="predictions/classifier/t5-large/${model}/${tag}"
    echo "  [${tag}/${model}] Routing (agreement gate + kappa)..."
    python classifier/postprocess/predict_complexity_kappa.py "${model}" \
        --use_agreement_gate \
        --tune_threshold \
        --valid_file "classifier/data/${DATASET}/${model}/binary_silver_single_vs_multi/train.json" \
        --output_path "${out}"
    echo "  [${tag}/${model}] Evaluating..."
    python evaluate_final_acc.py --pred_path "${out}"
}

declare -A BEST_CLF1_STD BEST_CLF2_STD

# =========================================================================
#  PHASE 0: Train standard Clf1 + Clf2
#  (shared by IT1, and as Clf2 for IT2–IT5, as Clf1 for IT6–IT7)
# =========================================================================

needs_std=false
for i in 1 2 3 4 5; do should_run "$i" && needs_std=true; done

if $needs_std; then
    if ! $SKIP_PHASE0; then
        echo ""
        echo "================================================================="
        echo "  PHASE 0: Training standard classifiers (Clf1 + Clf2)"
        echo "================================================================="

        cd "${CLF_DIR}"
        for m in "${MODELS[@]}"; do
            s=$(script_tag "$m")
            echo ""
            echo "--- Training standard Clf1 (no_ret_vs_ret) for ${m} ---"
            bash "run/run_large_train_${s}_no_ret_vs_ret.sh"
            echo ""
            echo "--- Training standard Clf2 (single_vs_multi) for ${m} ---"
            bash "run/run_large_train_${s}_single_vs_multi.sh"
        done
        cd "${REPO_ROOT}"
    else
        echo ""
        echo "================================================================="
        echo "  PHASE 0: Skipping training (SKIP_PHASE0=true)"
        echo "================================================================="
    fi

    echo ""
    echo "--- Selecting best epochs for standard classifiers ---"
    for m in "${MODELS[@]}"; do
        BEST_CLF1_STD[$m]=$(find_best_epoch \
            "classifier/outputs/${DATASET}/model/t5-large/${m}/no_ret_vs_ret/epoch" \
            "classifier/data/${DATASET}/${m}/silver/no_retrieval_vs_retrieval/valid.json")
        echo "  Clf1 ${m}: ${BEST_CLF1_STD[$m]}"

        BEST_CLF2_STD[$m]=$(find_best_epoch \
            "classifier/outputs/${DATASET}/model/t5-large/${m}/single_vs_multi/epoch" \
            "classifier/data/${DATASET}/${m}/silver/single_vs_multi/valid.json")
        echo "  Clf2 ${m}: ${BEST_CLF2_STD[$m]}"
    done
fi

# =========================================================================
#  IT1: Normal cascade
# =========================================================================
if should_run 1; then
    echo ""
    echo "================================================================="
    echo "  IT1: Normal cascade"
    echo "================================================================="
    for m in "${MODELS[@]}"; do
        route_split "iter1_standard" "$m" "${BEST_CLF1_STD[$m]}" "${BEST_CLF2_STD[$m]}"
    done
fi

# =========================================================================
#  IT2: Cascade w/ Clf1 undersampling
# =========================================================================
if should_run 2; then
    echo ""
    echo "================================================================="
    echo "  IT2: Clf1 undersampling"
    echo "================================================================="

    if ! $SKIP_TRAINING; then
        cd "${CLF_DIR}"
        for m in "${MODELS[@]}"; do
            s=$(script_tag "$m")
            echo "--- Training undersampled Clf1 for ${m} ---"
            bash "run/run_large_train_${s}_no_ret_vs_ret_undersampled.sh"
        done
        cd "${REPO_ROOT}"
    fi

    for m in "${MODELS[@]}"; do
        BEST_CLF1_UNDER=$(find_best_epoch \
            "classifier/outputs/${DATASET}/model/t5-large/${m}/no_ret_vs_ret_undersampled/epoch" \
            "classifier/data/${DATASET}/${m}/silver/no_retrieval_vs_retrieval/valid.json")
        echo "  Undersampled Clf1 (${m}): ${BEST_CLF1_UNDER}"
        route_split "iter2_undersampled" "$m" "${BEST_CLF1_UNDER}" "${BEST_CLF2_STD[$m]}"
    done
fi

# =========================================================================
#  IT3: Cascade w/ Clf1 weighted cross-entropy
# =========================================================================
if should_run 3; then
    echo ""
    echo "================================================================="
    echo "  IT3: Clf1 weighted cross-entropy"
    echo "================================================================="

    if ! $SKIP_TRAINING; then
        cd "${CLF_DIR}"
        for m in "${MODELS[@]}"; do
            s=$(script_tag "$m")
            echo "--- Training weighted CE Clf1 for ${m} ---"
            bash "run/run_large_train_${s}_no_ret_vs_ret_weighted_ce.sh"
        done
        cd "${REPO_ROOT}"
    fi

    for m in "${MODELS[@]}"; do
        BEST_CLF1_WCE=$(find_best_epoch \
            "classifier/outputs/${DATASET}/model/t5-large/${m}/no_ret_vs_ret_weighted_ce/epoch" \
            "classifier/data/${DATASET}/${m}/silver/no_retrieval_vs_retrieval/valid.json")
        echo "  Weighted CE Clf1 (${m}): ${BEST_CLF1_WCE}"
        route_split "iter3_weighted_ce" "$m" "${BEST_CLF1_WCE}" "${BEST_CLF2_STD[$m]}"
    done
fi

# =========================================================================
#  IT4: Cascade w/ Clf1 focal loss
# =========================================================================
if should_run 4; then
    echo ""
    echo "================================================================="
    echo "  IT4: Clf1 focal loss"
    echo "================================================================="

    if ! $SKIP_TRAINING; then
        cd "${CLF_DIR}"
        for m in "${MODELS[@]}"; do
            s=$(script_tag "$m")
            echo "--- Training focal Clf1 for ${m} ---"
            bash "run/run_large_train_${s}_no_ret_vs_ret_focal.sh"
        done
        cd "${REPO_ROOT}"
    fi

    for m in "${MODELS[@]}"; do
        BEST_CLF1_FOCAL=$(find_best_epoch \
            "classifier/outputs/${DATASET}/model/t5-large/${m}/no_ret_vs_ret_focal/epoch" \
            "classifier/data/${DATASET}/${m}/silver/no_retrieval_vs_retrieval/valid.json")
        echo "  Focal Clf1 (${m}): ${BEST_CLF1_FOCAL}"
        route_split "iter4_focal" "$m" "${BEST_CLF1_FOCAL}" "${BEST_CLF2_STD[$m]}"
    done
fi

# =========================================================================
#  IT5: Clf1 UE agreement gate (no Clf1 training)
# =========================================================================
if should_run 5; then
    echo ""
    echo "================================================================="
    echo "  IT5: Clf1 UE agreement gate"
    echo "================================================================="
    for m in "${MODELS[@]}"; do
        route_agreement "iter5_agreement" "$m" "${BEST_CLF2_STD[$m]}"
    done
fi

# =========================================================================
#  IT6: Clf2 κ(q) feature probing (analysis only, no training/routing)
# =========================================================================
if should_run 6; then
    echo ""
    echo "================================================================="
    echo "  IT6: Clf2 κ(q) feature probing"
    echo "================================================================="

    echo "--- Running κ(q) feature probe across all models ---"
    python classifier/postprocess/clf2_kappa_feature_probe.py --all_models \
        --output_dir "predictions/classifier/t5-large/iter6_kappa_probe"
fi

# =========================================================================
#  IT7: UE kappa (fully training-free)
# =========================================================================
if should_run 7; then
    echo ""
    echo "================================================================="
    echo "  IT7: UE kappa (fully training-free)"
    echo "================================================================="
    for m in "${MODELS[@]}"; do
        route_kappa "iter7_kappa" "$m"
    done
fi

# =========================================================================
echo ""
echo "================================================================="
echo "  All requested iterations complete."
echo "================================================================="
echo ""
echo "Results under predictions/classifier/t5-large/{model}/iter*/"
echo "Re-evaluate with:"
echo "  python evaluate_final_acc.py --pred_path predictions/classifier/t5-large/<model>/iter<N>_<tag>/"