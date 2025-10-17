import sys
import hydra
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
import time
from encoder_trainer import EncoderTrainer
from utils import seed_everything, setup_ddp, print_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # ✅ DDP (Distributed Data Parallel) setup
    if cfg.ddp.enabled:
        cfg.ddp.local_rank, cfg.ddp.global_rank = setup_ddp()
        print(f"DDP setup: local_rank={cfg.ddp.local_rank}, global_rank={cfg.ddp.global_rank}", flush=True)
    
    cfg.autoencoder.training.batch_size_per_gpu = cfg.autoencoder.training.batch_size // dist.get_world_size()
    if cfg.ddp.global_rank == 0:
        print_config(cfg)

    # ✅ Setup config
    cfg.autoencoder.model.checkpoints_prefix = cfg.autoencoder.model.checkpoints_prefix + \
        f"-num_latents={cfg.encoder.latent.num_latents}" \
        f"-{cfg.dataset.name}" \
        f"-{cfg.suffix}"

    # ✅ Initialize Weights and Biases
    if not cfg.ddp.enabled or dist.get_rank() == 0 and cfg.project.wandb_logging:
        name = cfg.autoencoder.model.checkpoints_prefix
        wandb.init(
            project=cfg.project.name,
            name=name,
            tags=[str(t) for t in cfg.project.tags],
            mode="online"
        )

    # ✅ Seed everything
    seed = cfg.project.seed + cfg.ddp.global_rank
    seed_everything(seed)

    # ✅ Initialize Trainer
    trainer = EncoderTrainer(cfg)
    trainer.train()

    # ✅ Destroy DDP process group
    if cfg.ddp.enabled:
        time.sleep(30)
        dist.barrier()
        dist.destroy_process_group() 


if __name__ == "__main__":
    # Filter out unrecognized arguments (like --local-rank)
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()



"""
HYDRA_FULL_ERROR=1 \
uv run \
torchrun --nproc_per_node=4 --master_port=12346 train_encoder.py \
dataset=wikipedia \
encoder.latent.num_latents=16 \
decoder.latent.num_latents=16 \
encoder.augmentation.masking.weight=0.5 \
encoder.augmentation.masking.encodings_mlm_probability=0.4 \
encoder.augmentation.gaussian_noise.weight=0.5 \
encoder.augmentation.gaussian_noise.delta=0.5 \
encoder.augmentation.latent_masking.probability=0.7 \
autoencoder.latent.dim=768 \
autoencoder.latent.num_latents=16 \
training.training_iters=100000 \
training="autoencoder" \
suffix="final"
"""




