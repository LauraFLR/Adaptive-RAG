#!/bin/bash
# Iteration 3b: Retrain Clf2 (B vs C) with structural feature prefix.
#
# Identical to run_large_train_{xl,xxl}_single_vs_multi.sh except:
#   - Training data:   binary_silver_feat_single_vs_multi/train.json  (feature-prefixed)
#   - Validation data: silver_feat_single_vs_multi/valid.json          (feature-prefixed)
#   - Predict data:    feat_predict.json                               (feature-prefixed)
#   - Output dir tag:  feat_single_vs_multi/
#
# Usage:
#   bash classifier/run/run_large_train_feat_single_vs_multi.sh flan_t5_xl
#   bash classifier/run/run_large_train_feat_single_vs_multi.sh flan_t5_xxl
#   bash classifier/run/run_large_train_feat_single_vs_multi.sh gpt

set -e

LLM_NAME=$1
if [ -z "$LLM_NAME" ]; then
    echo "Usage: $0 <flan_t5_xl|flan_t5_xxl|gpt>"
    exit 1
fi

DATE=feat
MODEL=t5-large
DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
GPU=0

echo "=== Iter 3b: Feature-augmented Clf2 for ${LLM_NAME} ==="

for EPOCH in 15 20 25 30 35
do
    echo ""
    echo "--- Epoch ${EPOCH} ---"

    # train
    TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/feat_single_vs_multi/epoch/${EPOCH}/${DATE}
    mkdir -p ${TRAIN_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${MODEL} \
        --train_file ./data/${DATASET_NAME}/${LLM_NAME}/binary_silver_feat_single_vs_multi/train.json \
        --question_column question \
        --answer_column answer \
        --labels B C \
        --learning_rate 3e-5 \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_train_batch_size 32 \
        --output_dir ${TRAIN_OUTPUT_DIR} \
        --overwrite_cache \
        --train_column 'train' \
        --do_train \
        --seed 42 \
        --num_train_epochs ${EPOCH}

    # valid (feature-prefixed validation)
    VALID_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/valid
    mkdir -p ${VALID_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/${LLM_NAME}/silver_feat_single_vs_multi/valid.json \
        --question_column question \
        --answer_column answer \
        --labels B C \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${VALID_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --do_eval

    # predict (feature-prefixed test set)
    PREDICT_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/predict
    mkdir -p ${PREDICT_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/feat_predict.json \
        --question_column question \
        --answer_column answer \
        --labels B C \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${PREDICT_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --do_eval
done

echo ""
echo "=== Done. Find outputs at: ==="
echo "  outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/feat_single_vs_multi/epoch/{15..35}/${DATE}/"
