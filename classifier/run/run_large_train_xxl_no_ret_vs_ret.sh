DATE=$(date +%Y_%m_%d)/$(date +%H_%M_%S)
MODEL=t5-large
LLM_NAME=flan_t5_xxl
DATASET_NAME=musique_hotpot_wiki2_nq_tqa_sqd
GPU=0

# Classifier 1: no-retrieval (A) vs retrieval (R = B+C merged)
# Training data:  silver/no_retrieval_vs_retrieval/train.json
# Validation data: silver/no_retrieval_vs_retrieval/valid.json

for EPOCH in 15 20 25 30 35
do
    # train
    TRAIN_OUTPUT_DIR=./outputs/${DATASET_NAME}/model/${MODEL}/${LLM_NAME}/no_ret_vs_ret/epoch/${EPOCH}/${DATE}
    mkdir -p ${TRAIN_OUTPUT_DIR}

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
        --train_column 'train' \
        --do_train \
        --seed 42 \
        --num_train_epochs ${EPOCH}


    # valid
    VALID_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/valid
    mkdir -p ${VALID_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/${LLM_NAME}/silver/no_retrieval_vs_retrieval/valid.json \
        --question_column question \
        --answer_column answer \
        --labels A R \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${VALID_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --do_eval


    # predict (on unlabeled test set)
    PREDICT_OUTPUT_DIR=${TRAIN_OUTPUT_DIR}/predict
    mkdir -p ${PREDICT_OUTPUT_DIR}

    CUDA_VISIBLE_DEVICES=${GPU} python run_classifier.py \
        --model_name_or_path ${TRAIN_OUTPUT_DIR} \
        --validation_file ./data/${DATASET_NAME}/predict.json \
        --question_column question \
        --answer_column answer \
        --labels A R \
        --max_seq_length 384 \
        --doc_stride 128 \
        --per_device_eval_batch_size 100 \
        --output_dir ${PREDICT_OUTPUT_DIR} \
        --overwrite_cache \
        --val_column 'validation' \
        --do_eval
done
