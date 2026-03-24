"""
Fine-tune diffusion model with SAE-filtered training targets (Scenario C).

Loads pretrained diffusion checkpoint and fine-tunes with the SAE applied
to the clean latent targets, so diffusion learns to predict SAE-reconstructed
latents directly.

Usage:
    python train_diffusion_sae.py \
        dataset=rocstories \
        encoder.latent.num_latents=16 \
        encoder.embedding.max_position_embeddings=128 \
        decoder.latent.num_latents=16 \
        decoder.embedding.max_position_embeddings=128 \
        autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"' \
        diffusion.model.load_checkpoint='"diffusion-rocstories-16-d=5-final/180000.pth"' \
        training=diffusion \
        suffix=sae-finetune \
        +sae_checkpoint=sae_checkpoints/topk_4x_k64/best.pt \
        +sae_type=topk \
        +sae_expansion_factor=4 \
        +sae_k=64 \
        +finetune_lr=1e-5 \
        +finetune_steps=20000
"""

import sys
import hydra
import torch
import torch.distributed as dist
import wandb
from omegaconf import DictConfig, OmegaConf
import time

from diffusion_trainer import DiffusionTrainer
from utils import seed_everything, print_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Force single-GPU
    cfg.ddp.enabled = False
    cfg.ddp.local_rank = 0
    cfg.ddp.global_rank = 0
    cfg.training = "diffusion"

    # Override training params for fine-tuning
    finetune_lr = cfg.get("finetune_lr", 1e-5)
    finetune_steps = cfg.get("finetune_steps", 20000)

    cfg.diffusion.optimizer.learning_rate = finetune_lr
    cfg.diffusion.optimizer.min_lr = finetune_lr
    cfg.diffusion.training.training_iters = finetune_steps
    cfg.diffusion.logging.eval_freq = min(5000, finetune_steps)
    cfg.diffusion.logging.save_freq = min(5000, finetune_steps)

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

    cfg.diffusion.training.batch_size_per_gpu = cfg.diffusion.training.batch_size

    print_config(cfg)
    seed_everything(cfg.project.seed)

    # Initialize trainer (will load pretrained checkpoint)
    trainer = DiffusionTrainer(cfg)

    # Reset optimizer LR to fine-tuning rate (checkpoint load overwrites it)
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = finetune_lr

    # Reset scheduler for fine-tuning
    trainer.step = 0
    trainer._setup_scheduler()

    # Load SAE for training-time target filtering
    sae_ckpt = cfg.get("sae_checkpoint", None)
    if sae_ckpt is None:
        raise ValueError("Must provide +sae_checkpoint=... for SAE fine-tuning")

    sae_type = cfg.get("sae_type", "topk")
    sae_expansion = cfg.get("sae_expansion_factor", 4)
    sae_k = cfg.get("sae_k", 64)

    trainer.load_sae(
        checkpoint_path=sae_ckpt,
        sae_type=sae_type,
        d_input=768,
        expansion_factor=sae_expansion,
        k=sae_k,
        use_at_inference=True,    # Also use at inference for consistency
        use_at_training=True,     # Key: filter training targets through SAE
    )

    print(f"Fine-tuning diffusion with SAE targets for {finetune_steps} steps "
          f"(lr={finetune_lr})")
    trainer.train()


if __name__ == "__main__":
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()
