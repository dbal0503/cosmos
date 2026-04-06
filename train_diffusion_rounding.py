"""
Fine-tune diffusion model with rounding-aware loss (Scenario D).

Adds a detached rounding penalty to the standard diffusion MSE loss.
The rounding target is computed by decoding the clean latent to tokens
via argmax, then re-encoding back to latent space. This directly teaches
the diffusion model to output latents that land on valid token-backed
representations.

Usage:
    python train_diffusion_rounding.py \
        dataset=rocstories \
        encoder.latent.num_latents=16 \
        encoder.embedding.max_position_embeddings=128 \
        decoder.latent.num_latents=16 \
        decoder.embedding.max_position_embeddings=128 \
        autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"' \
        diffusion.model.load_checkpoint='"diffusion-rocstories-16-d=5-final/180000.pth"' \
        diffusion.training.batch_size=64 \
        training=diffusion \
        suffix=rounding-finetune \
        +finetune_lr=1e-5 \
        +finetune_steps=20000 \
        +rounding_lambda=0.5
"""

import sys
import random
import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from typing import Dict, Tuple
from tqdm import trange

from diffusion_trainer import DiffusionTrainer
from utils import seed_everything, print_config, BatchEncoding


class RoundingAwareDiffusionTrainer(DiffusionTrainer):
    """
    DiffusionTrainer subclass that adds a rounding-aware loss term.

    Loss = MSE(f(x_t, t), x_0) + lambda * MSE(f(x_t, t), x_0_reencoded)

    where x_0_reencoded = Normalize(Encode(Argmax(Decode(Denorm(x_0)))))
    is the clean latent after a round-trip through decode/argmax/reencode.
    This is computed with no_grad — the re-encoded target is detached.
    """

    def __init__(self, cfg: DictConfig, rounding_lambda: float = 0.5):
        super().__init__(cfg)
        self.rounding_lambda = rounding_lambda
        self._rounding_targets_cache = {}

    @torch.no_grad()
    def _compute_rounding_target(self, clean_x: torch.Tensor) -> torch.Tensor:
        """
        Compute the re-encoded latent after round-trip through decoder.

        Args:
            clean_x: (B, N, d) normalized latents

        Returns:
            x_0_reencoded: (B, N, d) the token-backed latent target
        """
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Decode to logits
            pred_latents = self.autoencoder.denormalize_latent(clean_x)
            pred_logits = self.autoencoder.decoder(pred_latents)

            # Round: argmax to get token IDs
            rounded_ids = pred_logits.argmax(dim=-1)

            # Re-encode: token IDs -> BERT -> Perceiver -> normalize
            attention_mask = torch.ones_like(rounded_ids)
            re_batch = BatchEncoding({
                "input_ids": rounded_ids,
                "attention_mask": attention_mask,
            }).to(self.device)

            re_latents, _ = self.autoencoder.get_latent(
                re_batch, bert_output_masking=False
            )
            x_0_reencoded = self.autoencoder.normalize_latent(re_latents)

        return x_0_reencoded

    def calc_loss(self, batch) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Compute diffusion loss with rounding-aware penalty.

        Loss = MSE(pred, clean_x) + lambda * MSE(pred, x_0_reencoded)
        """
        from utils.diffusion_utils import get_stat

        # Get latent (same as parent)
        batch = batch.to(self.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            encoder_latents, _ = self.autoencoder.get_latent(
                batch, bert_output_masking=False
            )
            clean_x = self.autoencoder.normalize_latent(encoder_latents)

        # Compute rounding target (detached, no grad)
        # Move encoder/decoder to device if needed
        self.autoencoder.encoder.to(self.device)
        self.autoencoder.decoder.to(self.device)
        x_0_reencoded = self._compute_rounding_target(clean_x)

        # Add noise to the clean latent
        batch_size = clean_x.size(0)
        t = self.sample_time(batch_size)
        marg_forward = self.dynamic.marginal(clean_x, t)
        x_t, noise = marg_forward['x_t'], marg_forward['noise']

        # Self-conditioning
        x_0_self_cond = torch.zeros_like(clean_x, dtype=clean_x.dtype)
        if self.cfg.diffusion.diffusion.use_self_cond and random.random() > 0.5:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                x_0_self_cond = self.ddp_score_estimator(
                    x_t=x_t.clone(),
                    time_t=t.clone(),
                    x_0_self_cond=x_0_self_cond
                ).detach()

        # Model prediction
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            x_0_pred = self.ddp_score_estimator(
                x_t=x_t,
                time_t=t,
                x_0_self_cond=x_0_self_cond
            )

        # Standard diffusion loss: predict the clean latent
        loss_standard = torch.mean(torch.square(clean_x - x_0_pred))

        # Rounding-aware loss: also attract predictions toward token-backed latents
        loss_rounding = torch.mean(torch.square(x_0_reencoded - x_0_pred))

        # Combined loss
        total_loss = loss_standard + self.rounding_lambda * loss_rounding

        # Statistics
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            # Compute actual rounding error of the prediction for monitoring
            rounding_error = torch.norm(
                clean_x.float() - x_0_reencoded.float(), dim=-1
            ).mean()

            loss_dict = {
                "loss_standard": {"mean": loss_standard.item()},
                "loss_rounding": {"mean": loss_rounding.item()},
                "rounding_error": {"mean": rounding_error.item()},
                "clean_x": get_stat(clean_x.detach()),
                "x_0": get_stat(x_0_pred.detach()),
                "x_t": get_stat(x_t.detach()),
                "x_0_reencoded": get_stat(x_0_reencoded.detach()),
            }

        return total_loss, loss_dict


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
    rounding_lambda = cfg.get("rounding_lambda", 0.5)

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

    # Initialize trainer
    trainer = RoundingAwareDiffusionTrainer(cfg, rounding_lambda=rounding_lambda)

    # Reset optimizer LR
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = finetune_lr

    # Reset scheduler for fine-tuning
    trainer.step = 0
    trainer._setup_scheduler()

    print(f"Fine-tuning diffusion with rounding-aware loss for {finetune_steps} steps "
          f"(lr={finetune_lr}, lambda={rounding_lambda})")
    trainer.train()


if __name__ == "__main__":
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()
