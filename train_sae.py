"""
Train Sparse Autoencoder (SAE) on cached latents from Cosmos autoencoder.

Supports TopK SAE and Dense AE (ReLU) baseline.
Designed for single-GPU training on Colab.

Usage:
    python train_sae.py \
        --latents_dir cached_latents/rocstories \
        --sae_type topk \
        --expansion_factor 4 \
        --k 64 \
        --lr 3e-4 \
        --num_steps 50000 \
        --batch_size 4096 \
        --output_dir sae_checkpoints/topk_4x_k64

    # Dense AE ablation:
    python train_sae.py \
        --latents_dir cached_latents/rocstories \
        --sae_type dense \
        --expansion_factor 4 \
        --lr 3e-4 \
        --output_dir sae_checkpoints/dense_4x
"""

import argparse
import os
import json
import time
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange

from architecture.sparse_autoencoder import TopKSparseAutoencoder, DenseAutoencoder


def load_cached_latents(latents_dir: str, split: str = "train") -> torch.Tensor:
    """Load all cached latent chunks from disk."""
    split_dir = os.path.join(latents_dir, split)
    files = sorted(glob.glob(os.path.join(split_dir, "latents_*.pt")))
    if not files:
        raise FileNotFoundError(f"No latent files found in {split_dir}")

    chunks = []
    for f in files:
        chunk = torch.load(f, map_location="cpu")
        chunks.append(chunk.float())  # Convert from bfloat16 to float32

    latents = torch.cat(chunks, dim=0)
    print(f"Loaded {latents.shape[0]} latents from {len(files)} files, "
          f"shape: {latents.shape}")
    return latents


def create_sae(sae_type: str, d_input: int, expansion_factor: int, k: int):
    """Create SAE model based on type."""
    if sae_type == "topk":
        return TopKSparseAutoencoder(
            d_input=d_input,
            expansion_factor=expansion_factor,
            k=k,
        )
    elif sae_type == "dense":
        return DenseAutoencoder(
            d_input=d_input,
            expansion_factor=expansion_factor,
        )
    else:
        raise ValueError(f"Unknown SAE type: {sae_type}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load data
    train_latents = load_cached_latents(args.latents_dir, "train")
    # Shape: (N_samples, N_latents=16, d=768) -> flatten to per-vector
    n_samples, n_latents, d_input = train_latents.shape
    train_flat = train_latents.reshape(-1, d_input)  # (N_samples * 16, 768)
    print(f"Flattened: {train_flat.shape[0]} vectors of dim {d_input}")

    # Load validation latents if available
    val_latents = None
    try:
        val_latents = load_cached_latents(args.latents_dir, "test")
        val_flat = val_latents.reshape(-1, d_input)
        print(f"Validation: {val_flat.shape[0]} vectors")
    except FileNotFoundError:
        print("No validation latents found, skipping validation.")

    # Create DataLoader
    train_dataset = TensorDataset(train_flat)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    sae = create_sae(args.sae_type, d_input, args.expansion_factor, args.k)
    sae.to(device)
    print(f"SAE parameters: {sum(p.numel() for p in sae.parameters()):,}")
    print(f"SAE hidden dim: {sae.d_hidden}, k={args.k if args.sae_type == 'topk' else 'N/A'}")

    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    sae.train()
    step = 0
    data_iter = iter(train_loader)
    log_history = []
    best_fvu = float("inf")
    start_time = time.time()

    # Track dead features with EMA over a window (more reliable than per-batch)
    feature_activation_ema = torch.zeros(sae.d_hidden, device=device)
    ema_decay = 0.999

    pbar = trange(1, args.num_steps + 1, desc="Training SAE")
    for step in pbar:
        # Get batch (cycle through data)
        try:
            (batch_z,) = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            (batch_z,) = next(data_iter)

        batch_z = batch_z.to(device)

        # Forward + loss (compute_loss returns codes for dead feature tracking)
        loss, info, s = sae.compute_loss(batch_z)

        # Update dead feature tracking using codes from compute_loss
        with torch.no_grad():
            batch_alive = (s != 0).any(dim=0).float()  # (d_hidden,)
            feature_activation_ema = ema_decay * feature_activation_ema + (1 - ema_decay) * batch_alive
            windowed_dead_frac = (feature_activation_ema < 1e-3).float().mean().item()

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize decoder weights (for TopK SAE)
        if hasattr(sae, "normalize_decoder_weights"):
            sae.normalize_decoder_weights()

        # Logging
        if step % args.log_every == 0:
            elapsed = time.time() - start_time
            log_entry = {
                "step": step,
                "mse": info["mse"],
                "fvu": info["fvu"],
                "l0": info["l0"],
                "dead_frac_batch": info["dead_frac"],
                "dead_frac_ema": windowed_dead_frac,
                "elapsed_s": elapsed,
            }
            log_history.append(log_entry)

            pbar.set_description(
                f"MSE={info['mse']:.6f} FVU={info['fvu']:.4f} "
                f"L0={info['l0']:.1f} Dead(ema)={windowed_dead_frac:.3f}"
            )

        # Validation
        if step % args.eval_every == 0 and val_latents is not None:
            sae.eval()
            val_losses = []
            val_fvus = []
            with torch.no_grad():
                # Evaluate on random subset
                n_val = min(args.batch_size * 10, val_flat.shape[0])
                idx = torch.randperm(val_flat.shape[0])[:n_val]
                val_batch = val_flat[idx].to(device)

                val_loss, val_info, _ = sae.compute_loss(val_batch)
                val_losses.append(val_info["mse"])
                val_fvus.append(val_info["fvu"])

            avg_val_fvu = sum(val_fvus) / len(val_fvus)
            print(f"\n[Step {step}] Val MSE={sum(val_losses)/len(val_losses):.6f} "
                  f"Val FVU={avg_val_fvu:.4f}")

            if avg_val_fvu < best_fvu:
                best_fvu = avg_val_fvu
                save_checkpoint(sae, optimizer, step, args, "best.pt")
                print(f"  -> New best FVU: {best_fvu:.4f}")

            sae.train()

        # Checkpoint
        if step % args.checkpoint_every == 0:
            save_checkpoint(sae, optimizer, step, args, f"step_{step}.pt")

    # Final checkpoint
    save_checkpoint(sae, optimizer, step, args, "final.pt")

    # Save training log
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    print(f"Training complete. Best FVU: {best_fvu:.4f}")
    print(f"Checkpoints saved to {args.output_dir}")


def save_checkpoint(sae, optimizer, step, args, filename):
    """Save SAE checkpoint."""
    save_path = os.path.join(args.output_dir, filename)
    torch.save({
        "sae_state_dict": sae.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": vars(args),
    }, save_path)


def main():
    parser = argparse.ArgumentParser(description="Train SAE on cached Cosmos latents")
    parser.add_argument("--latents_dir", type=str, required=True,
                        help="Directory with cached latents (from extract_latents.py)")
    parser.add_argument("--sae_type", type=str, default="topk", choices=["topk", "dense"],
                        help="SAE type: topk or dense (ReLU baseline)")
    parser.add_argument("--expansion_factor", type=int, default=4,
                        help="Hidden dim = d_input * expansion_factor")
    parser.add_argument("--k", type=int, default=64,
                        help="TopK sparsity (only for topk type)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Batch size (number of latent vectors)")
    parser.add_argument("--num_steps", type=int, default=50000,
                        help="Number of training steps")
    parser.add_argument("--log_every", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--eval_every", type=int, default=1000,
                        help="Evaluate on validation set every N steps")
    parser.add_argument("--checkpoint_every", type=int, default=2000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--output_dir", type=str, default="sae_checkpoints/default",
                        help="Output directory for checkpoints and logs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
