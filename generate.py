import sys
import hydra
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
import time
import json
import os
import torch

from diffusion_trainer import DiffusionTrainer
from utils import seed_everything, setup_ddp, print_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # DDP setup (only if enabled)
    if cfg.ddp.enabled:
        cfg.ddp.local_rank, cfg.ddp.global_rank = setup_ddp()
        cfg.diffusion.training.batch_size_per_gpu = cfg.diffusion.training.batch_size // dist.get_world_size()
        if cfg.ddp.global_rank == 0:
            print_config(cfg)
    else:
        # Single-GPU mode
        cfg.ddp.local_rank = 0
        cfg.ddp.global_rank = 0
        cfg.diffusion.training.batch_size_per_gpu = cfg.diffusion.training.batch_size
        print_config(cfg)

    # Seed everything
    seed = cfg.project.seed + cfg.ddp.global_rank
    seed_everything(seed)

    # Initialize Trainer
    trainer = DiffusionTrainer(cfg)
    trainer.restore_checkpoint()
    trainer.estimate()


if __name__ == "__main__":
    # Filter out unrecognized arguments (like --local-rank)
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()

"""
# DDP mode (original):
CUDA_LAUNCH_BLOCKING=1 \
HYDRA_FULL_ERROR=1 \
uv run \
torchrun --nproc_per_node=4 --master_port=12345 \
generate.py \
dataset=rocstories \
diffusion.dynamic.N=200 \
diffusion.dynamic.d=5 \
diffusion.training.batch_size=512 \
encoder.latent.num_latents=16 \
encoder.embedding.max_position_embeddings=128 \
decoder.latent.num_latents=16 \
decoder.embedding.max_position_embeddings=128 \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"' \
diffusion.model.load_checkpoint='"diffusion-rocstories-16-d=5-final/180000.pth"' \
diffusion.generation.num_gen_texts=2000 \
training=""

# Single-GPU mode:
HYDRA_FULL_ERROR=1 python generate.py \
ddp.enabled=false \
dataset=rocstories \
diffusion.dynamic.N=200 \
diffusion.dynamic.d=5 \
diffusion.training.batch_size=64 \
encoder.latent.num_latents=16 \
encoder.embedding.max_position_embeddings=128 \
decoder.latent.num_latents=16 \
decoder.embedding.max_position_embeddings=128 \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"' \
diffusion.model.load_checkpoint='"diffusion-rocstories-16-d=5-final/180000.pth"' \
diffusion.generation.num_gen_texts=100 \
training=""
"""
