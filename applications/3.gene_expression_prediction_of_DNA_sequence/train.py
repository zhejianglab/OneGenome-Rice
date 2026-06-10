# Standard library
import os
import argparse
import json

# Third-party libraries
import wandb
import pandas as pd
import torch
from functools import partial

# Hugging Face Transformers
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments
)

# Local imports
from src.util import (
    dist_print,
    is_main_process,
    setup_distributed,
    setup_logging,
    setup_seed,
    get_index,
    setup_sync_batchnorm
)
from src.dataset import  MultiTrackDataset
from src.model import GenOmics, targets_scaling_torch
from src.metrics import compute_multimodal_metrics
from src.trainer import(
    CustomTrainer, 
    DistributedSamplerCallback, 
    LocalLoggerCallback
    )

    
def parse_args():
    """
    Parse CLI arguments and return an ``args`` object.
    """
    parser = argparse.ArgumentParser(description="Train RNA-seq track predictor with configurable args.")

    # --- Data paths ---
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained model (Hugging Face format).")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Path to the tokenizer (Hugging Face format).")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Checkpoint directory to resume training from.")
    parser.add_argument("--use_flash_attn", action='store_true',
                        help="Enable FlashAttention acceleration (default: disabled).")
    parser.add_argument("--sequence_split_train", type=str, 
                        help="Training split index file.")
    parser.add_argument("--sequence_split_train_multi", type=str, nargs='+',
                        help="Training split index files (multiple).")
    parser.add_argument("--sequence_split_val", type=str, required=True, 
                        help="Validation split index file.")
    parser.add_argument("--index_stat_json", type=str, 
                        help="Training data statistics JSON (index_stat.json).")
    parser.add_argument("--index_stat_multi_json", type=str, nargs='+',
                        help="Multiple training data statistics JSONs (index_stat.json).")
    parser.add_argument("--nonzero_means",type=float,nargs='+',
                        help="Per-track non-zero mean values.")
    # --- Output settings ---
    parser.add_argument("--output_base_dir", type=str, required=True,
                        help="Base output directory.")

    # Debugging / convenience
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Debug: limit number of training samples (None means no limit).")
    parser.add_argument("--max_sequence_length", type=int, default=32768)

    # --- Chromosome splits ---
    parser.add_argument("--train_chromosomes", type=str, nargs='+', default=["chr19"],
                        help="List of chromosomes used for training.")
    parser.add_argument("--val_chromosomes", type=str, nargs='+', default=["Chr12"],
                        help="List of chromosomes used for validation.")

    # --- Training hyperparameters ---
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--batch_size_per_device", type=int, default=1,
                        help="Per-GPU batch size.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                        help="Number of dataloader workers.")
    parser.add_argument("--gpus_per_node", type=int, default=8,
                        help="Number of GPUs per node.")
    # --- Model settings ---
    parser.add_argument("--loss_func", type=str, default="mse",
                        choices=["mse", "poisson", "tweedie", "poisson-multinomial"], 
                        help="Loss function type.")
    parser.add_argument("--proj_dim", type=int, default=1024,
                        help="U-Net input feature dimension.")
    parser.add_argument("--num_downsamples", type=int, default=4,
                        help="Number of downsampling blocks in the U-Net.")
    parser.add_argument("--bottleneck_dim", type=int, default=1536,
                        help="U-Net bottleneck dimension.")
    
    # --- Misc ---
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging.")
    parser.add_argument("--seed", type=int, default=42,
                    help="Random seed.")

    return parser.parse_args()

def main():
    """
    Main training entrypoint: fine-tune a pretrained DNA language model with multi-track BigWig
    signals for single-base-resolution prediction.

    Supports Distributed Data Parallel (DDP), optional FlashAttention-2 + bf16 acceleration,
    and Weights & Biases logging.
    """

    # Parse arguments
    args  = parse_args()

    # Set random seed
    setup_seed(args.seed)


    # Initialize variables to avoid locals() issues
    train_dataset = None
    val_dataset = None
    run = None
    
    # --- Distributed init ---
    local_rank, world_size, is_distributed = setup_distributed()
    
    # Logging
    log_filepath = setup_logging(
        output_base_dir=args.output_base_dir,
    )
    dist_print(f"[DDP] Distributed initialization complete: local_rank={local_rank}, world_size={world_size}")

    # Weights & Biases
    if args.use_wandb and is_main_process():
        wandb_config = {
                "learning_rate": args.lr,
                "batch_size": args.batch_size_per_device,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "epochs": args.num_train_epochs,
                "model": args.model_path,
                "loss_func": args.loss_func,
                "max_sequence_length": args.max_sequence_length,
                "proj_dim": args.proj_dim,
                "bottleneck_dim": args.bottleneck_dim,
                "num_downsamples": args.num_downsamples,
                "use_flash_attn": args.use_flash_attn,
                "seed": args.seed,
                "train_chromosomes": args.train_chromosomes,
                "val_chromosomes": args.val_chromosomes,
                }
        run = wandb.init(
                entity="zhongliyuan-bgi-group",
                project="RNA-seq",  
                name=f"train-{args.loss_func}-lr{args.lr}-bs{args.batch_size_per_device}",
                dir=args.output_base_dir,
                resume="allow",
                config=wandb_config)
        # Define metric summaries
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("eval/loss", summary="min")
        wandb.define_metric("epoch")
        wandb.define_metric("global_step")
        
        dist_print(f"[wandb] Logged in as: {run.entity}")
        dist_print(f"[wandb] Project: {run.project} | Run Name: {run.name}")
        dist_print(f"[wandb] Run URL: {run.url}")
        dist_print(f"[wandb] Local Dir: {run.dir}")
    
    # Print args
    args_dict = vars(args)
    dist_print("[config] Training configuration:")
    for key, value in args_dict.items():
        dist_print(f"    {key}: {value}")

    # --- Load model and tokenizer ---
    dist_print("[load] Loading pretrained model and tokenizer...")
    if args.use_flash_attn:
        dist_print("[perf] Using FlashAttention")
        base_model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            revision="main",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16  # Use bf16 weights
        )
    else:
        base_model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            revision="main",
            torch_dtype=torch.bfloat16  # Use bf16 weights
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_dir,
        trust_remote_code=True,
        revision="main",
        padding_side='right',
    )


    # --- Data split ---
    dist_print(f"[data] Train chromosomes: {args.train_chromosomes}")
    dist_print(f"[data] Validation chromosomes: {args.val_chromosomes}")

    # --- Load split index ---
    dist_print("[data] Loading split index...")
    if args.sequence_split_train is not None:
            train_index_df = get_index(args.sequence_split_train)
    # --- Filter training index by chromosome ---
            selected_train_index_df = train_index_df[train_index_df["chromosome"].str.extract(r'(Chr\d+)')[0].isin(args.train_chromosomes)].copy()
            if args.max_train_samples is not None:
                selected_train_index_df = selected_train_index_df.sample(n=args.max_train_samples, random_state=args.seed)
    elif args.sequence_split_train_multi is not None:
            train_indexes = args.sequence_split_train_multi
            train_index_dfs = [get_index(index_df) for index_df in train_indexes]
            selected_train_index_df=[]
            for train_index_df in train_index_dfs:
                temp_df = train_index_df[train_index_df["chromosome"].str.extract(r'(Chr\d+)')[0].isin(args.train_chromosomes)].copy()
                if args.max_train_samples is not None:
                    temp_df = temp_df[:args.max_train_samples]
                selected_train_index_df.append(temp_df)
    else:
        raise ValueError("You must provide either --sequence_split_train or --sequence_split_train_multi.")

    val_index_df = get_index(args.sequence_split_val)


    # --- Index filtering ---
    #selected_train_index_df = train_index_df[train_index_df["chromosome"].isin(args.train_chromosomes)].copy()
    # Chromosomes are already defined in run_sequence_split_and_meta_extract.py, so no need to filter again here.

    # selected_val_index_df = val_index_df[val_index_df["chromosome"].isin(args.val_chromosomes)].copy()
    # if args.max_train_samples is not None:
    #     selected_val_index_df = selected_train_index_df

    # --- Load index statistics ---

    if args.index_stat_json is not None:
        with open(args.index_stat_json, "r") as f:
            index_stat = json.load(f)
    elif args.index_stat_multi_json is not None:
        index_stat_jsons = args.index_stat_multi_json
        index_stat = []
        for index_stat_json in index_stat_jsons:
            with open(index_stat_json, "r") as f:
                temp_index_stat = json.load(f)
            index_stat.append(temp_index_stat)
    else:
        raise ValueError("You must provide either --index_stat_json or --index_stat_multi_json.")

    
    # --- Build datasets ---
    dist_print("[data] Building training dataset...")
    train_dataset = MultiTrackDataset(selected_train_index_df, index_stat, 
                                      tokenizer, max_length=args.max_sequence_length)
    dist_print(f"[data] Train: {len(train_dataset):,} samples")
    # dist_print("[data] Building validation dataset...")
    # val_dataset = MultiTrackDataset(selected_train_index_df, label_meta_df, 
    #                                 index_stat, tokenizer, max_length=args.max_sequence_length)
                                      
    # dist_print(f"[data] Validation: {len(val_dataset):,} samples")
    

    if args.index_stat_multi_json is not None:
        temp = index_stat[0]
        index_stat=temp
        index_stat['counts']['nonzero_mean']=[]
        for non0_mean in args.nonzero_means:
            index_stat['counts']['nonzero_mean'].append(non0_mean)

    # --- Build downstream predictor ---
    dist_print("[model] Building downstream network...")
    model = GenOmics(
        base_model,
        index_stat=index_stat,
        loss_func=args.loss_func,
        proj_dim=args.proj_dim,
        num_downsamples=args.num_downsamples,
        bottleneck_dim=args.bottleneck_dim
    )
    
    # --- SyncBatchNorm ---
    model = setup_sync_batchnorm(model, is_distributed, args.gpus_per_node)
    dist_print("[ddp] SyncBatchNorm configured")
    
    # --- Cast to bfloat16 ---
    model = model.to(torch.bfloat16)
    dist_print("[perf] BF16 enabled")

    
    # # --- Optionally freeze the backbone and unfreeze only the last layer ---
    # for param in model.base.parameters():
    #     param.requires_grad = False
    # dist_print("[model] Freezing all base model parameters")
    # for param in model.base.layers[-1].parameters():
    #     param.requires_grad = True
    # dist_print("[model] Unfreezing the last layer")


    # --- Parameter counts ---
    trainable_base_params = sum(p.numel() for p in model.base.parameters() if p.requires_grad)
    total_base_params = sum(p.numel() for p in model.base.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    downstread_task_head_params = total_params - total_base_params
    
    dist_print(
        f"[model] Total parameters: {total_params:,} "
        f"(downstream head: {downstread_task_head_params:,}, "
        f"trainable base ratio: {trainable_base_params/total_base_params*100:.1f}%)"
    )

    # --- Training arguments ---
    dist_print("[train] Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_base_dir,
        logging_dir=os.path.join(args.output_base_dir, "logs"),

        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_device,
        per_device_eval_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        
        dataloader_num_workers = args.dataloader_num_workers,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        include_for_metrics=["inputs", "loss"],

        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adafactor",

        # eval_strategy="epoch",
        save_strategy="epoch",
        # eval_accumulation_steps=10,
        save_total_limit=30,
        save_safetensors=True,

        fp16=False,
        bf16=True,
        half_precision_backend="auto",

        logging_steps=1,
        report_to="none",
        log_level="info",

        # ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        seed=args.seed,

        resume_from_checkpoint=args.ckpt_dir,
    )
    
    # --- Trainer ---
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        compute_metrics=partial(compute_multimodal_metrics, val_chromosomes=args.val_chromosomes, tokenizer=tokenizer),
        callbacks=[DistributedSamplerCallback(),
        LocalLoggerCallback(log_file_path=log_filepath)]
    )
    try:
        # --- Train ---
        dist_print("[train] Starting training...")
        if args.ckpt_dir: 
            # Resume training
            trainer.train(resume_from_checkpoint=args.ckpt_dir)
        else:
            trainer.train()
        dist_print("[train] Training complete!")

    except Exception as e:
        dist_print(f"[error] Training failed: {str(e)}")
        if torch.distributed.is_initialized():
            torch.distributed.barrier()  # Prevent other ranks from hanging
        raise  # Re-raise


    finally:
        # Cleanup datasets
        dataset_dict = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset
        }

        for name, ds in dataset_dict.items():
            if ds is not None and hasattr(ds, 'close'):
                ds.close()
                dist_print(f"[cleanup] Released resources: {name} ({type(ds).__name__})")

        # Cleanup W&B
        if run is not None and is_main_process():
            wandb.finish()
            dist_print("[wandb] Run finished")

    dist_print("[done] Main process finished!")


if __name__ == "__main__":
    main()
