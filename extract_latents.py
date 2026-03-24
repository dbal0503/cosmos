"""
Extract and cache normalized latents from the frozen Cosmos autoencoder.

Saves latent tensors to disk so SAE training doesn't require the full
autoencoder pipeline on each epoch.

Usage:
    python extract_latents.py \
        dataset=rocstories \
        encoder.latent.num_latents=16 \
        encoder.embedding.max_position_embeddings=128 \
        decoder.latent.num_latents=16 \
        decoder.embedding.max_position_embeddings=128 \
        autoencoder.model.load_checkpoint='"autoencoder-num_latents=16-wikipedia-final-128/100000.pth"'
"""

import sys
import os
import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoder_trainer import EncoderTrainer
from utils import seed_everything, BatchEncoding
from utils.sharded_dataset import ShardedDataset
from utils.pylogger import RankedLogger


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Force single-GPU, no training mode
    cfg.ddp.enabled = False
    cfg.ddp.local_rank = 0
    cfg.ddp.global_rank = 0
    cfg.training = ""

    seed_everything(cfg.project.seed)
    logger = RankedLogger(name="extract_latents", rank_zero_only=False, rank=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Output directory
    output_dir = os.path.join(cfg.project.path, "cached_latents", cfg.dataset.name)
    os.makedirs(output_dir, exist_ok=True)

    # Load autoencoder
    logger.info("Loading autoencoder...")
    autoencoder = EncoderTrainer(cfg)
    autoencoder.encoder.eval()
    autoencoder.decoder.eval()
    autoencoder.encoder.to(device)

    # Verify latent statistics are loaded from checkpoint
    if not hasattr(autoencoder, 'latent_mean') or autoencoder.latent_mean is None:
        raise RuntimeError(
            "Autoencoder checkpoint must contain latent_mean and latent_std. "
            "Ensure the checkpoint path is correct and the checkpoint was saved "
            "after computing latent statistics."
        )

    # Tokenizer for collation
    tokenizer = autoencoder.tokenizer

    def collate_fn(batch):
        texts = [sample["text_trg"] for sample in batch]
        tokenized = tokenizer(
            texts,
            add_special_tokens=cfg.tokenizer.add_special_tokens,
            padding=cfg.tokenizer.padding,
            truncation=cfg.tokenizer.truncation,
            max_length=cfg.dataset.max_sequence_len,
            return_tensors=cfg.tokenizer.return_tensors,
            return_attention_mask=cfg.tokenizer.return_attention_mask,
            return_token_type_ids=cfg.tokenizer.return_token_type_ids,
        )
        return BatchEncoding({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "text_trg": texts,
        })

    # Process each split
    for split in ["train", "test"]:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        dataset_path = os.path.join(cfg.dataset.dataset_path, split)
        if not os.path.exists(dataset_path):
            logger.info(f"Skipping {split}: {dataset_path} does not exist")
            continue

        logger.info(f"Extracting latents for {split} split...")

        sharded = ShardedDataset(cfg, split, prefetch_shards=1, logger=logger)

        all_latents = []
        batch_count = 0
        chunk_idx = 0
        chunk_size = 10000  # Save in chunks to manage memory

        while True:
            shard = sharded.get_next_shard()
            if shard is None:
                break

            loader = DataLoader(
                shard,
                batch_size=cfg.get("extract_batch_size", 64),
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=min(4, os.cpu_count() or 1),
                drop_last=False,
            )

            for batch in tqdm(loader, desc=f"{split} shard"):
                batch = batch.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                    encoder_latents, _ = autoencoder.get_latent(batch, bert_output_masking=False)
                    normalized = autoencoder.normalize_latent(encoder_latents)

                all_latents.append(normalized.cpu().float())
                batch_count += normalized.shape[0]

                # Save chunk when we've accumulated enough
                if batch_count >= chunk_size:
                    chunk_tensor = torch.cat(all_latents, dim=0)
                    save_path = os.path.join(split_dir, f"latents_{chunk_idx:04d}.pt")
                    torch.save(chunk_tensor, save_path)
                    logger.info(f"Saved {chunk_tensor.shape[0]} latents to {save_path}")

                    all_latents = []
                    batch_count = 0
                    chunk_idx += 1

            # Clean up shard
            del shard, loader
            torch.cuda.empty_cache()

        # Save remaining latents
        if all_latents:
            chunk_tensor = torch.cat(all_latents, dim=0)
            save_path = os.path.join(split_dir, f"latents_{chunk_idx:04d}.pt")
            torch.save(chunk_tensor, save_path)
            logger.info(f"Saved {chunk_tensor.shape[0]} latents to {save_path}")

        sharded.stop()
        logger.info(f"Done extracting {split} latents.")

    # Also save latent statistics for convenience
    stats_path = os.path.join(output_dir, "latent_stats.pt")
    torch.save({
        "latent_mean": autoencoder.latent_mean.cpu(),
        "latent_std": autoencoder.latent_std.cpu(),
    }, stats_path)
    logger.info(f"Saved latent statistics to {stats_path}")


if __name__ == "__main__":
    sys.argv = [arg for arg in sys.argv if not arg.startswith("--local-rank")]
    main()
