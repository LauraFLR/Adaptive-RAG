#!/usr/bin/env bash

DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=t5-large
LLM_NAME=gpt
DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
GPU=${GPU:-0}

# Weighted CE via FocalLossTrainer with gamma=0.
FOCAL_GAMMA=${FOCAL_GAMMA:-0.0}
FOCAL_ALPHA=${FOCAL_ALPHA:-0.71}

for EPOCH in 35 40
do
	TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/no_ret_vs_ret_weighted_ce/epoch/${EPOCH}/${DATE}
	mkdir -p ${TRAIN_OUTPUT_DIR}

	echo "[TRAIN] ${LLM_NAME} epoch=${EPOCH} gamma=${FOCAL_GAMMA} alpha=${FOCAL_ALPHA}"
	CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
		--model_name_or_path ${MODEL} \
		--train_file ./data/${DATASET_NAME}/${LLM_NAME}/silver/no_retrieval_vs_retrieval/train.json \
		--question_column question \
		--answer_column answer \
		--labels A R \
		--learning_rate 3e-5 \
		--max_seq_length 384 \
		--doc_stride 128 \
		--per_device_train_batch_size 32 \
		--output_dir ${TRAIN_OUTPUT_DIR} \
		--overwrite_cache \
		--train_column train \
		--do_train \
		--seed 42 \
		--num_train_epochs ${EPOCH} \
		--use_focal_loss \
		--focal_gamma ${FOCAL_GAMMA} \
		--focal_alpha ${FOCAL_ALPHA}

	CKPT_PATH=$(ls -d ${TRAIN_OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -V | tail -n 1)
	if [[ -z "${CKPT_PATH}" ]]; then
		CKPT_PATH=${TRAIN_OUTPUT_DIR}
	fi

	# Clean up extra checkpoints
	for ckpt in ${TRAIN_OUTPUT_DIR}/checkpoint-*; do
		if [[ -d "${ckpt}" && "${ckpt}" != "${CKPT_PATH}" ]]; then
			rm -rf "${ckpt}"
		fi
	done

	VALID_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/valid
	mkdir -p ${VALID_OUTPUT_DIR}
	echo "[VALID] ${LLM_NAME} epoch=${EPOCH} checkpoint=${CKPT_PATH}"
	CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
		--model_name_or_path ${CKPT_PATH} \
		--validation_file ./data/${DATASET_NAME}/${LLM_NAME}/silver/no_retrieval_vs_retrieval/valid.json \
		--question_column question \
		--answer_column answer \
		--labels A R \
		--max_seq_length 384 \
		--doc_stride 128 \
		--per_device_eval_batch_size 100 \
		--output_dir ${VALID_OUTPUT_DIR} \
		--overwrite_cache \
		--val_column validation \
		--do_eval || echo "[WARN] Validation failed for epoch ${EPOCH}"

	PREDICT_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/predict
	mkdir -p ${PREDICT_OUTPUT_DIR}
	echo "[PREDICT] ${LLM_NAME} epoch=${EPOCH} checkpoint=${CKPT_PATH}"
	CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
		--model_name_or_path ${CKPT_PATH} \
		--validation_file ./data/${DATASET_NAME}/predict.json \
		--question_column question \
		--answer_column answer \
		--labels A R \
		--max_seq_length 384 \
		--doc_stride 128 \
		--per_device_eval_batch_size 100 \
		--output_dir ${PREDICT_OUTPUT_DIR} \
		--overwrite_cache \
		--val_column validation \
		--do_eval || echo "[WARN] Prediction failed for epoch ${EPOCH}"

	echo "[COMPLETE] Epoch ${EPOCH}"
done