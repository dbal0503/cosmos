"""
Single-GPU entry point for Cosmos.
Bypasses DDP/torchrun and runs directly on a single GPU.

Usage:
    python run_single_gpu.py mode=generate dataset=rocstories \
        encoder.latent.num_latents=16 \
        encoder.embedding.max_position_embeddings=128 \
        decoder.latent.num_latents=16 \
        decoder.embedding.max_position_embeddings=128 \
        autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"' \
        diffusion.model.load_checkpoint='"diffusion-rocstories-16-d=5-final/180000.pth"' \
        diffusion.generation.num_gen_texts=100

Modes:
    generate   - Generate text with pretrained model
    train_diff - Train diffusion model
"""

import sys
import os
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from utils import seed_everything, print_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Force single-GPU mode
    cfg.ddp.enabled = False
    cfg.ddp.local_rank = 0
    cfg.ddp.global_rank = 0

    # Set batch size per gpu = total batch size (single GPU)
    if cfg.training == "diffusion":
        cfg.diffusion.training.batch_size_per_gpu = cfg.diffusion.training.batch_size
    elif cfg.training == "autoencoder":
        cfg.autoencoder.training.batch_size_per_gpu = cfg.autoencoder.training.batch_size

    print_config(cfg)
    seed_everything(cfg.project.seed)

    mode = cfg.get("mode", "generate")

    if mode == "generate":
        from diffusion_trainer import DiffusionTrainer
        trainer = DiffusionTrainer(cfg)
        trainer.restore_checkpoint()
        trainer.estimate()

    elif mode == "train_diff":
        from diffusion_trainer import DiffusionTrainer

        # Setup checkpoint prefix
        cfg.diffusion.model.checkpoints_prefix = (
            cfg.diffusion.model.checkpoints_prefix
            + f"-{cfg.dataset.name}"
            + f"-{cfg.encoder.latent.num_latents}"
            + f"-d={cfg.diffusion.dynamic.d}"
            + f"-{cfg.diffusion.architecture.unconditional_encoder.num_hidden_layers}-layers"
        )
        if cfg.suffix:
            cfg.diffusion.model.checkpoints_prefix += f"-{cfg.suffix}"

        trainer = DiffusionTrainer(cfg)
        trainer.train()

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'generate' or 'train_diff'.")


if __name__ == "__main__":
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()
