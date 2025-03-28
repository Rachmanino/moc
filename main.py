import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import default_data_collator

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from pretraining_utils import training_utils, args_utils
from pretraining_utils.dataloader import PreprocessedIterableDataset

import datetime, pdb, pickle
from torch.profiler import profile, ProfilerActivity

transformers.logging.set_verbosity_error()

# Import models from different methods
from transformers import LlamaForCausalLM, LlamaConfig
from cola import ColaConfig, ColaForCausalLM, ColaMForCausalLM
from moc.modeling_llama_moc import LlamaForCausalLM_MoC
import bitsandbytes as bnb
from galore import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor

# Import extra utilities
from pretraining_utils.dist_utils import get_rank, is_main_process
from pretraining_utils.seed_utils import seed_everything
seed_everything(42) # For reproducibility

# Enable FA
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

# The path to c4/en dataset
c4_en_path = '/data/datasets/c4/en'

# The method to choose from 
model_choices = ['llama', 'sltrain', 'colam', 'moc']

def parse_args(args):
    parser = argparse.ArgumentParser()

    # For building corresponding models
    parser.add_argument(
        "--model_type", type=str, required=True, choices=model_choices
    )
    # For loading corresponding model configurations
    parser.add_argument("--model_config", type=str, required=True)

    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="AdamW") # Use AdamW as default
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", 
                        type=str, 
                        default="cosine", 
                        choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_training_steps", 
                        type=int, 
                        default=10_000,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, 
                        default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)

    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    # GaLore parameters
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--update_proj_gap", type=int, default=50)
    parser.add_argument("--galore_scale", type=float, default=1.0)
    parser.add_argument("--proj_type", type=str, default="std")

    # SLTrain parameters
    parser.add_argument("--peft_model", type=str, default='full', choices=model_choices)
    parser.add_argument("--lora_alpha", type=float, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.)
    parser.add_argument("--train_scaling", default=False, action="store_true")
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--sp_ratio", type=float, default=0.01)

    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    # val_data = datasets.load_dataset("allenai/c4", "en", split="validation", streaming=True) #DGX
    val_data = datasets.load_dataset(c4_en_path, split="validation", streaming=True) #DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        loss = model(**batch, labels=labels).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens

def main(args):
    raise NotImplementedError("This script is not ready for use yet.")

if __name__ == "__main__":
    print('Successfully import all necessary modules!') # For environment check
    raise NotImplementedError("This script is not ready for use yet.")
    print("Starting script")
    args = parse_args(None)
    main(args)