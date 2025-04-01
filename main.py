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

# Enable FA
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

c4_en_path = '/data/datasets/c4/en' # The path to c4/en dataset

# The method to choose from 
model_choices = ['llama', 'colam', 'moc']

def parse_args(args):
    parser = argparse.ArgumentParser()

    # For building corresponding models
    parser.add_argument(
        "--model_type", type=str, required=True, choices=model_choices
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="moc_pretrain")
    # For loading corresponding model configurations
    parser.add_argument("--model_config", type=str, required=True)

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
    parser.add_argument("--max_train_tokens", 
                        type=training_utils.max_train_tokens_to_number, 
                        default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
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

    # # SLTrain parameters
    # parser.add_argument("--lora_alpha", type=float, default=32)
    # parser.add_argument("--lora_dropout", type=float, default=0.)
    # parser.add_argument("--train_scaling", default=False, action="store_true")
    # parser.add_argument("--rank", type=int, default=128)
    # parser.add_argument("--sp_ratio", type=float, default=0.01)

    args = parser.parse_args(args)
    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(
    model, 
    preprocess_batched, 
    pad_idx, 
    global_rank, 
    world_size, 
    device, 
    batch_size
):
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
    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID")))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    dist.init_process_group(
        backend="nccl",
        rank=global_rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=3600),
    )

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            if is_main_process():
                logger.info(
                    f"{args.gradient_accumulation}-{world_size}-{args.total_batch_size}-{args.batch_size}"
                )
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # turn off logger
    if global_rank != 0:
        logger.remove() # Only rank 0 will log

    if global_rank == 0:
        model_name = args.model_config.split("/")[1]
        model_name = model_name.split(".")[0]
        run_name = (
            model_name
            if args.run_name is None
            else args.run_name
        )
        wandb.init(project=args.wandb_project, name=run_name)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)
    data = datasets.load_dataset(c4_en_path, split="train", streaming=True)

    seed_for_shuffle = 42
    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)

    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data,
            rank=global_rank,
            world_size=world_size,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=args.max_length
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice

    dataset = PreprocessedIterableDataset(
        data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=args.workers
    )
    eval_dataloader = None

    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=args.max_length
    )

    #TODO Build model
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")

        if args.model_type.lower() == "colam":
            model_config = ColaConfig.from_pretrained(args.continue_from)
            model = ColaMForCausalLM.from_pretrained(
                args.continue_from, torch_dtype=torch.bfloat16
            ).to(device=device)
        elif args.model_type.lower() == "moc":
            model_config = LlamaConfig.from_pretrained(args.continue_from)
            model = LlamaForCausalLM_MoC.from_pretrained(
                args.continue_from, torch_dtype=torch.bfloat16
            ).to(device=device)
        else:
            model_config = AutoConfig.from_pretrained(args.continue_from)
            model = LlamaForCausalLM.from_pretrained(
                args.continue_from, torch_dtype=torch.bfloat16
            ).to(device=device)
        logger.info(f"Model successfully loaded")
    
    else:
        logger.info(
            f"Training from scratch, global step will start from zero"
        )
        logger.info("*" * 40)

        if args.model_type.lower() == "colam":
            model_config = ColaConfig.from_pretrained(args.model_config)
            model = ColaMForCausalLM(model_config)
        elif args.model_type.lower() == "moc":
            model_config = LlamaConfig.from_pretrained(args.model_config)
            model = LlamaForCausalLM_MoC(model_config)
        else:
            model_config = AutoConfig.from_pretrained(args.model_config)
            model = LlamaForCausalLM(model_config)

        if args.dtype in ["bf16", "bfloat16"]:
            model = model.to(device=device, dtype=torch.bfloat16)
        else:
            model = model.to(device=device)

    # Enable GCP
    if args.activation_checkpointing:
        if args.model_type.lower() == 'colam':
            logger.warning(f"Already using checkpointing in CoLA-M, ignored")
        else:
            model.gradient_checkpointing_enable()

    if 'galore' in args.optimizer.lower():
        # make parameters with "rank" to a single group, if param_name has "mlp" or "attn"
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            
            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': args.rank, 'update_proj_gap': args.update_proj_gap, 'scale': args.galore_scale, 'proj_type': args.proj_type}]

    if 'galore' in args.optimizer.lower():
        logger.info(f"Total params with GaLore enabled: {sum(p.numel() for p in galore_params) / 1_000_000:.2f}M")
    
    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # ====== starting config ======= #
    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    layer_wise_flag = True if "per_layer" in args.optimizer.lower() else False

    if args.optimizer.lower() == "galore_adamw":
        optimizer = GaLoreAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = training_utils.build_optimizer(model, trainable_params, args)
        if layer_wise_flag:
            if not isinstance(optimizer, dict):
                raise ValueError("Layer-wise optimizer is not properly constructed.")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if args.continue_from is not None:
        optimizer_checkpoint = torch.load(
            os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu"
        )
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        logger.info(f"Optimizer and scheduler restored from {args.continue_from}")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.jo  in(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )

    scheduler_start_step = update_step

    # print params and trainable params
    logger.info(f"Running with {args.model_type}\n")
    logger.info(f"\n{model}\n")
    logger.info(
        f"All params: \n{[n for n,p in model.named_parameters() if p.requires_grad]}\n"
    )
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Total non-low-rank parameters: "
        f"{sum(p.numel() for n,p in model.named_parameters() if 'cola_' not in n) / 1_000_000:.2f}M"
    )
    if "cola" in args.model_type.lower():
        logger.info(
            f"Total low-rank parameters: "
            f"{sum(p.numel() for n, p in model.named_parameters() if 'cola_' in n) / 1_000_000:.2f}M"
        )
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "allenai/c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script
        pbar = tqdm(
            total=args.num_training_steps - update_step, desc="Update steps", ncols=80
        )

    if not args.single_gpu:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # ##############################
    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Maximum memory allocated before training: {max_memory} bytes\n")
    torch.cuda.reset_peak_memory_stats()
        
    for batch_idx, batch in enumerate(train_dataloader):
        if batch_idx // args.gradient_accumulation < update_step:
            # Skipping data that are already seen in previous steps
            continue

        global_step += 1
        local_step += 1

        if update_step > args.num_training_steps:
            logger.info(
                f"Reached max number of update steps (f{args.num_training_steps}). Stopping training."
            )
            print(f"Rank {global_rank} stopping training.")
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = (
            batch["input_ids"].clone() if "labels" not in batch else batch["labels"]
        )
        batch["labels"][batch["labels"] == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

        loss = model(**batch).loss
        scaled_loss = loss / args.gradient_accumulation

        scaled_loss.backward()

        if global_step % args.gradient_accumulation != 0:
            continue

        #######
        if args.grad_clipping != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)

        grad_norm = sum(
            [
                torch.norm(p.grad.clone().detach().cpu())
                for p in model.parameters()
                if p.grad is not None
            ]
        )

        if global_rank == 0:
            pbar.update(1)

        if not layer_wise_flag:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        update_step += 1
        update_time = time.time() - update_time

        # save checkpoint by save_every
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
            and global_rank == 0
        ):
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            logger.info(
                f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            model.module.save_pretrained(
                current_model_directory, max_shard_size="100GB"
            )

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
                "tokens_seen_before": tokens_seen_before,
                "update_time": update_time,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            model.eval()
            total_loss, evaluated_on_tokens = evaluate_model(
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
            )
            if global_rank == 0:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_perplexity": np.exp(total_loss),
                        "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(
                f"Eval loss and ppl. at step {update_step}: {total_loss}, {np.exp(total_loss)}"
            )
            model.train()

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        if global_rank == 0:
            wandb.log(
                {
                    "loss": loss.item(),
                    "lr": lr,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                    "gradnorm": grad_norm,
                    "max_memory": max_memory,
                },
                step=update_step,
            )

        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0:
        pbar.close()

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    if global_rank == 0 and not os.path.exists(current_model_directory):
        logger.info(
            f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(args.save_dir, exist_ok=True)

        model.module.save_pretrained(current_model_directory)

        optimizer_checkpoint = {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
    )

    if global_rank == 0:
        wandb.log(
            {
                "final_eval_loss": total_loss,
                "final_eval_perplexity": np.exp(total_loss),
                "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(
            f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
        )

    logger.info("Script finished successfully")

if __name__ == "__main__":
    # print('Successfully import all necessary modules!') # For environment check
    # raise NotImplementedError("This script is not ready for use yet.")
    print("Starting script")
    args = parse_args(None)
    seed_everything(args.seed) # For reproducibility
    main(args)