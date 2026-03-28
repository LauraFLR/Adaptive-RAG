#!/bin/bash
# Retrain Clf2 (B vs C) using ONLY silver prediction-derived labels
# (no binary inductive-bias labels from dataset origin)
#
# Key difference from original training:
#   Original:    binary_silver_single_vs_multi/train.json  (~3268 samples, silver + inductive-bias)
#   Silver-only: silver/single_vs_multi/train.json         (~868 samples, prediction-derived only)

set -e

DATE=silver_only
MODEL=t5-large
LLM_NAME=$1
DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
GPU=0

if [ -z "$LLM_NAME" ]; then
    echo "Usage: $0 <flan_t5_xl|flan_t5_xxl>"
    exit 1
fi

echo "=== Retraining Clf2 (silver-only) for ${LLM_NAME} ==="
echo "Training data: silver/single_vs_multi/train.json"

for EPOCH in 15 20 25 30 35
do
    echo ""
    echo "--- Epoch ${EPOCH} ---"

    TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/single_vs_multi/epoch/${EPOCH}/${DATE}
    mkdir -p ${TRAIN_OUTPUT_DIR}

    # Train
    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${MODEL} \
        --train_file ./data/${DATASET_NAME}/${LLM_NAME}/silver/single_vs_multi/train.json \
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
        --num_train_epochs ${EPOCH}

    # Validate
    VALID_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/valid
    mkdir -p ${VALID_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/${LLM_NAME}/silver/single_vs_multi/valid.json \
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

    # Predict on test questions
    PREDICT_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/predict
    mkdir -p ${PREDICT_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/predict.json \
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
echo "=== All epochs done for ${LLM_NAME} ==="
echo "Outputs at: outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/single_vs_multi/epoch/*/silver_only/"
