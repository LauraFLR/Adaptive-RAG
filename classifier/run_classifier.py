#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library's seq2seq models for question answering using the 🤗 Seq2SeqTrainer.
"""
# You can also adapt this script on your own question answering task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
os.environ['TRANSFORMERS_CACHE'] = os.path.dirname(os.getcwd()) + '/cache'
import random
from pathlib import Path
from typing import List, Optional, Tuple
import copy
#from utils_qa import *
import pickle

import datasets
import nltk
import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    SchedulerType,
    Trainer,
    TrainingArguments,
    get_scheduler,
)
from transformers.utils.versions import require_version
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction, get_last_checkpoint


from torch.nn import CrossEntropyLoss

from utils import *


class FocalLoss(torch.nn.Module):
    """Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Args:
        gamma (float): Focusing parameter. 0 reduces to cross-entropy.
        alpha: Class weights. Accepts:
            - ``None``                     — no alpha weighting;
            - ``float``                    — weight for index 1 (minority class);
              internally stored as ``[1-alpha, alpha]`` for binary tasks;
            - ``list`` / ``torch.Tensor``  — explicit per-class weights.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, float):
            self.alpha = torch.tensor([1.0 - alpha, alpha], dtype=torch.float)
        elif isinstance(alpha, (list, torch.Tensor)):
            self.alpha = (
                torch.tensor(alpha, dtype=torch.float)
                if not isinstance(alpha, torch.Tensor)
                else alpha.float().clone().detach()
            )
        else:
            raise TypeError(
                f"alpha must be None, a float, a list, or a torch.Tensor; got {type(alpha)}"
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute mean focal loss.

        Args:
            logits:  Raw (un-normalised) class scores, shape ``(N, C)``.
            targets: Ground-truth class indices, shape ``(N,)``, dtype long.
        """
        probs = torch.softmax(logits, dim=-1).clamp(min=1e-8)
        batch_idx = torch.arange(probs.size(0), device=logits.device)
        p_t = probs[batch_idx, targets]
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = -focal_weight * torch.log(p_t)
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = alpha_t * loss
        return loss.mean()


class FocalLossTrainer(Trainer):
    """HuggingFace Trainer subclass that replaces cross-entropy with FocalLoss.

    Extra constructor kwargs:
        focal_loss_fn (FocalLoss): Pre-built FocalLoss instance.
        label_token_ids (list[int]): Vocabulary token IDs for each class label,
            in the same order as ``args.labels``.  Used to select the relevant
            logit columns from the seq2seq decoder output at position 0.
    """

    def __init__(self, *args, focal_loss_fn: FocalLoss, label_token_ids: list, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn
        self.label_token_ids = label_token_ids

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        """Override compute_loss to apply FocalLoss on the first decoder position."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        # outputs.logits: (batch, tgt_seq_len, vocab_size)
        # For a single-token class label the relevant score is at position 0.
        logits = outputs.logits[:, 0, :]  # (batch, vocab_size)

        # Narrow to the num_labels columns that correspond to the class tokens.
        tid = torch.tensor(self.label_token_ids, dtype=torch.long, device=logits.device)
        label_logits = logits[:, tid]  # (batch, num_labels)

        # Convert vocab token IDs stored in labels[:,0] → 0-based class indices.
        token_ids = labels[:, 0]  # (batch,)
        class_indices = (token_ids.unsqueeze(1) == tid.unsqueeze(0)).long().argmax(dim=1)

        loss = self.focal_loss_fn(label_logits, class_indices)
        return (loss, outputs) if return_outputs else loss


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/question-answering/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

##
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1":
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# option_to_label and label_to_option are built dynamically in main() from --labels.

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a QA task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=384,
        help=(
            "The maximum total input sequence length after "
            "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
        ),
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text (useful for T5 models).",
    )

    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--do_eval", action="store_true", help="To do eval on the question answering model")
    parser.add_argument("--do_train", action="store_true", help="To do train on the question answering model")
    # data col
    parser.add_argument(
        "--train_column",
        type=str,
        default='train',
        help="The name of the train column in the datasets.",
    )
    parser.add_argument(
        "--val_column",
        type=str,
        default='validation',
        help="The name of the validation column in the datasets.",
    )
    parser.add_argument(
        "--test_column",
        type=str,
        default='test',
        help="The name of the test column in the datasets.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    parser.add_argument(
        "--max_answer_length",
        type=int,
        default=30,
        help=(
            "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        ),
    )
    parser.add_argument(
        "--val_max_answer_length",
        type=int,
        default=None,
        help=(
            "The maximum total sequence length for validation "
            "target text after tokenization.Sequences longer than this will be truncated, sequences shorter will be "
            "padded. Will default to `max_target_length`.This argument is also used to override the ``max_length`` "
            "param of ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help=(
            "Number of beams to use for evaluation. This argument will be "
            "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--question_column",
        type=str,
        default='question',
        help="The name of the column in the datasets containing the questions (for question answering).",
    )
    parser.add_argument(
        "--answer_column",
        type=str,
        default='answers',
        help="The name of the column in the datasets containing the answers (for question answering).",
    )

    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=["A", "B", "C"],
        help="Ordered list of label tokens the classifier predicts. E.g. 'A B C' (3-class), 'A R' (no-ret vs ret), 'B C' (single vs multi).",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Replace cross-entropy with Focal Loss to handle class imbalance.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focusing parameter gamma for Focal Loss (default: 2.0). Higher values down-weight easy examples more strongly.",
    )
    parser.add_argument(
        "--focal_alpha",
        type=float,
        default=None,
        help=(
            "Scalar alpha weight for the minority class (index 1). "
            "Converted to a [1-alpha, alpha] per-class tensor before being passed to FocalLoss. "
            "Example: --focal_alpha 0.75 gives weights [0.25, 0.75]."
        ),
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        default=128,
        help="When splitting up a long document into chunks how much stride to take between chunks.",
    )
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    label_to_option = {i: label for i, label in enumerate(args.labels)}

    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    device = accelerator.device

    if args.source_prefix is None and args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )


    # Make one log on every process with the configuration for debugging.
    # TODO
    # Setup logging
    logging.basicConfig(        
        filename=args.output_dir+'/logs.log', # 
        filemode='w',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True
    )

    #logger.info(accelerator.state, main_process_only=False)
    logger.info(accelerator.state)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    logger.info(args)


    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.do_eval:
            extension = args.validation_file.split(".")[-1]
        else:
            extension = args.train_file.split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)
        
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.


    # load model and tokenizer
    model, tokenizer = load_model(args)

        
    if args.do_train:
        if args.train_column not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets[args.train_column]

        if args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(args.max_train_samples))


        # Create train feature from dataset
        with accelerator.main_process_first():
            train_dataset = train_dataset.map(
                preprocess_features_function, 
                fn_kwargs={'args':args, 'raw_datasets':raw_datasets, 'tokenizer': tokenizer},
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            if args.max_train_samples is not None:
                # Number of samples might increase during Feature Creation, We select only specified max samples
                train_dataset = train_dataset.select(range(args.max_train_samples))
    
    
    if args.do_eval:
        if args.val_column not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets[args.val_column]

        if args.max_eval_samples is not None:
            # We will select sample from whole data
            eval_examples = eval_examples.select(range(args.max_eval_samples))
        # Validation Feature Creation
        with accelerator.main_process_first():
            eval_dataset = eval_examples.map(
                preprocess_features_function, 
                fn_kwargs={'args':args, 'raw_datasets':raw_datasets, 'tokenizer': tokenizer},
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=eval_examples.column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

        if args.max_eval_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.mixed_precision == "fp16" else None,
    )

    if args.do_train:
        train_dataset_for_model = train_dataset.remove_columns(["example_id", "offset_mapping"])
        train_dataloader = DataLoader(
            train_dataset_for_model, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )

    if args.do_eval:
        eval_dataset_for_model = eval_dataset.remove_columns(["example_id", "offset_mapping"])        
        eval_dataloader = DataLoader(
            eval_dataset_for_model, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size
        )


    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)


    # Prepare everything with our `accelerator`.
    # FocalLossTrainer manages its own device placement internally, so skip
    # accelerator wrapping for model/optimizer/training-dataloader when focal
    # loss is enabled.
    if not args.use_focal_loss:
        model, optimizer = accelerator.prepare(
            model, optimizer
        )

    if args.do_train and not args.use_focal_loss:
        train_dataloader = accelerator.prepare(
            train_dataloader
        )

    if args.do_eval:
        eval_dataloader = accelerator.prepare(
            eval_dataloader
        )

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("no_trainer", experiment_config)

    # Train!
    if args.do_train and args.use_focal_loss:
        # ── Focal-Loss training via FocalLossTrainer (HF Trainer subclass) ──────
        if args.focal_alpha is not None:
            _alpha_tensor = torch.tensor(
                [1.0 - args.focal_alpha, args.focal_alpha], dtype=torch.float
            )
        else:
            _alpha_tensor = None

        _label_token_ids = [tokenizer(lbl).input_ids[0] for lbl in args.labels]
        _focal_loss_fn = FocalLoss(gamma=args.focal_gamma, alpha=_alpha_tensor)

        _fl_training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed if args.seed is not None else 42,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.num_warmup_steps,
            save_strategy="epoch",
            eval_strategy="no",
            logging_dir=args.output_dir,
            report_to="none",
        )

        focal_trainer = FocalLossTrainer(
            model=model,
            args=_fl_training_args,
            train_dataset=train_dataset_for_model,
            data_collator=data_collator,
            focal_loss_fn=_focal_loss_fn,
            label_token_ids=_label_token_ids,
        )
        focal_trainer.train()
        model = focal_trainer.model

    if args.do_train and not args.use_focal_loss:
        
        args.max_train_steps, args.num_train_epochs, lr_scheduler_train = prepare_scheduler(args, accelerator, train_dataloader, optimizer, args.max_train_steps, args.num_train_epochs)

        total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
                accelerator.load_state(args.resume_from_checkpoint)
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            total_loss = 0
            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch:
                    if resume_step is not None and step < resume_step:
                        completed_steps += 1
                        continue

                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler_train.step()
                    optimizer.zero_grad()

                    # logger.info("Loss:{} ".format(loss))

                    # We keep track of the loss at each epoch
                    total_loss = total_loss + loss.cpu().detach().float()

                logger.info(tokenizer.batch_decode(batch["input_ids"][:1], skip_special_tokens=True))

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps }"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if completed_steps >= args.max_train_steps:
                    break

            logger.info("Epoch %d Loss:{} ".format(total_loss / len(train_dataloader)), epoch) 

            if args.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)

            if args.push_to_hub and epoch < args.num_train_epochs - 1:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    repo.push_to_hub(
                        commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                    )


            if args.output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(args.output_dir)
                    if args.push_to_hub:
                        repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)



    # Validation
    if args.do_eval:
        logger.info("***** Running Validation *****")
        logger.info(f"  Num examples = {len(eval_dataset)}")
        logger.info(f"  Batch size = {args.per_device_eval_batch_size}")

        if args.val_max_answer_length is None:
            args.val_max_answer_length = args.max_answer_length

        gen_kwargs = {
            "max_length": args.val_max_answer_length,
            #'no_repeat_ngram_size':2
            #"num_beams": args.num_beams,
        }

        # inference
        model.eval()
        predictions = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():

                scores = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    return_dict_in_generate=True,
                    output_scores=True,
                    **gen_kwargs,
                ).scores[0]

                probs = (
                    torch.nn.functional.softmax(
                        torch.stack([
                            scores[:, tokenizer(label).input_ids[0]]
                            for label in args.labels
                        ]), dim=0,
                    ).detach().cpu().numpy()
                )

                preds_labels = np.argmax(probs, 0)
                preds = [label_to_option[pred] for pred in preds_labels]

                labels = batch["labels"]
                labels = accelerator.gather_for_metrics(labels)
                labels = labels.cpu().numpy()

                predictions = predictions + preds

                if args.ignore_pad_token_for_loss:
                    # Replace -100 in the labels as we can't decode them.
                    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)


                logger.info('==========================================')
                logger.info(tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True))
                logger.info('Prediction : ')
                logger.info(preds)
                logger.info('Answer : ')
                logger.info(tokenizer.batch_decode(labels, skip_special_tokens=False))


        gold_answers = eval_examples['answer']

        dict_id_pred_results = {qid : {'prediction': pred, 'answer' : ans, 'dataset_name' : data} for qid, pred, ans, data in zip(eval_examples['id'], predictions, gold_answers, eval_examples['dataset_name'])}
        with open(os.path.join(args.output_dir, "dict_id_pred_results.json"), "w") as f:
            json.dump(dict_id_pred_results, f, indent=4)

        assert len(gold_answers) == len(predictions)


        final_acc_score = calculate_accuracy(gold_answers, predictions)
        final_eval_results = {'final_acc_score' : final_acc_score}

        logger.info(f"Evaluation metrics: {final_eval_results}")
        print(final_eval_results)

        with open(os.path.join(args.output_dir, "final_eval_results.json"), "w") as f:
            json.dump(final_eval_results, f)

        # Acc per class
        final_eval_results_perClass = calculate_accuracy_perClass(gold_answers, predictions, labels=args.labels)

        logger.info(f"Evaluation metrics per class: {final_eval_results_perClass}")
        print(final_eval_results_perClass)

        with open(os.path.join(args.output_dir, "final_eval_results_perClass.json"), "w") as f:
            json.dump(final_eval_results_perClass, f, indent=4)


if __name__ == "__main__":
    main()





