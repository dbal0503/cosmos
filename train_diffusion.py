import sys
import hydra
import torch.distributed as dist
import wandb
from omegaconf import DictConfig
import time

from diffusion_trainer import DiffusionTrainer
from utils import seed_everything, setup_ddp, print_config


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # ✅ DDP (Distributed Data Parallel) setup
    if cfg.ddp.enabled:
        cfg.ddp.local_rank, cfg.ddp.global_rank = setup_ddp()
    
    cfg.diffusion.training.batch_size_per_gpu = cfg.diffusion.training.batch_size // dist.get_world_size()
    if cfg.ddp.global_rank == 0:
        print_config(cfg)

    # ✅ Setup config
    cfg.diffusion.model.checkpoints_prefix = cfg.diffusion.model.checkpoints_prefix + \
        f"-{cfg.dataset.name}" \
        f"-{cfg.encoder.latent.num_latents}" \
        f"-d={cfg.diffusion.dynamic.d}" \
        f"-{cfg.diffusion.architecture.unconditional_encoder.num_hidden_layers}-layers"
    if cfg.suffix:
        cfg.diffusion.model.checkpoints_prefix += f"-{cfg.suffix}"
    
    # ✅ Initialize Weights and Biases
    if not cfg.ddp.enabled or dist.get_rank() == 0 and cfg.project.wandb_logging:
        name = cfg.diffusion.model.checkpoints_prefix
        wandb.init(
            project=cfg.project.name,
            name=name,
            tags=[str(t) for t in cfg.project.tags],
            mode="online",
        )

    # ✅ Seed everything
    seed = cfg.project.seed + cfg.ddp.global_rank
    seed_everything(seed)

    # ✅ Initialize Trainer
    trainer = DiffusionTrainer(cfg)
    trainer.train()

    # ✅ Destroy DDP process group
    if cfg.ddp.enabled:
        time.sleep(300)
        dist.barrier()
        dist.destroy_process_group() 


if __name__ == "__main__":
    # Filter out unrecognized arguments (like --local-rank)
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()


"""
CUDA_LAUNCH_BLOCKING=1 \
HYDRA_FULL_ERROR=1 \
uv run \
torchrun --nproc_per_node=4 --master_port=12345 \
train_diffusion.py \
dataset=rocstories \
diffusion.dynamic.N=200 \
diffusion.dynamic.d=5 \
diffusion.training.batch_size=512 \
encoder.latent.num_latents=16 \
encoder.embedding.max_position_embeddings=128 \
decoder.latent.num_latents=16 \
decoder.embedding.max_position_embeddings=128 \
autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"' \
diffusion.generation.num_gen_texts=2000 \
training=diffusion \
suffix="final"
"""

