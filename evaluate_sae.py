"""
Evaluate SAE integration with Cosmos diffusion pipeline.

Compares baseline (no SAE), SAE-filtered, and Dense AE-filtered generation
on MAUVE, diversity, PPL, and rounding error metrics.

Usage:
    python evaluate_sae.py \
        --dataset rocstories \
        --autoencoder_ckpt "autoencoder-num_latents=16-wikipedia-final-128/100000.pth" \
        --diffusion_ckpt "diffusion-rocstories-16-d=5-final/180000.pth" \
        --sae_ckpt sae_checkpoints/topk_4x_k64/best.pt \
        --sae_type topk \
        --expansion_factor 4 \
        --k 64 \
        --num_texts 2000 \
        --output_dir evaluation_results
"""

import argparse
import os
import sys
import json

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from architecture.sparse_autoencoder import TopKSparseAutoencoder, DenseAutoencoder


def compute_rounding_error(pred_embeddings: torch.Tensor,
                           autoencoder, device: torch.device) -> dict:
    """
    Compute rounding error: L2 distance between predicted latents and
    the re-encoded latents after rounding through the decoder.

    The "round trip" is: pred_latents -> decode to logits -> argmax tokens
    -> re-encode -> re-compress -> re-normalize. The L2 distance between
    the original predicted latents and the re-encoded ones measures how
    far the diffusion output is from a valid (token-backed) latent.

    Args:
        pred_embeddings: (B, N, d) predicted latent embeddings (normalized space)
        autoencoder: the EncoderTrainer instance (has encoder, decoder, tokenizer,
                     normalize_latent, denormalize_latent, get_latent)
        device: torch device

    Returns:
        dict with rounding error statistics
    """
    all_l2 = []
    chunk_size = 64  # process in chunks to manage memory

    for i in range(0, pred_embeddings.shape[0], chunk_size):
        chunk = pred_embeddings[i:i + chunk_size].to(device)  # (chunk, N, d)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
            # Decode predicted latents to logits
            pred_latents = autoencoder.denormalize_latent(chunk)
            pred_logits = autoencoder.decoder(pred_latents)

            # Round: argmax to get token IDs
            rounded_ids = pred_logits.argmax(dim=-1)  # (chunk, seq_len)

            # Re-encode: token IDs -> BERT -> Perceiver -> normalize
            # Build a minimal batch for the encoder
            attention_mask = torch.ones_like(rounded_ids)
            from utils import BatchEncoding
            re_batch = BatchEncoding({
                "input_ids": rounded_ids,
                "attention_mask": attention_mask,
            })
            re_latents, _ = autoencoder.get_latent(re_batch, bert_output_masking=False)
            re_normalized = autoencoder.normalize_latent(re_latents)

        # L2 distance per sample
        l2 = torch.norm(chunk.float() - re_normalized.float(), dim=-1).mean(dim=-1)  # (chunk,)
        all_l2.append(l2.cpu())

    all_l2 = torch.cat(all_l2)

    return {
        "rounding_error_mean": all_l2.mean().item(),
        "rounding_error_std": all_l2.std().item(),
        "rounding_error_median": all_l2.median().item(),
        "rounding_error_max": all_l2.max().item(),
    }


def evaluate_model(trainer, sae, use_sae, num_texts, device):
    """Run generation and compute all metrics for a single model variant."""
    # Configure SAE usage
    if sae is not None and use_sae:
        trainer.sae = sae
        trainer.use_sae_inference = True
    else:
        trainer.sae = None
        trainer.use_sae_inference = False

    # Generate texts and collect embeddings
    trainer.autoencoder.decoder.to(device)
    all_gen_texts = []
    all_trg_texts = []
    all_pred_embeddings = []

    trainer._setup_valid_data_generator()

    for batch in trainer.valid_loader:
        batch = batch.to(device)

        gen_text, pred_emb, _ = trainer.generate_text_batch(
            batch_size=len(batch["input_ids"])
        )

        all_gen_texts += gen_text
        all_trg_texts += batch["text_trg"]
        all_pred_embeddings.append(pred_emb.cpu())

        if len(all_gen_texts) >= num_texts:
            break

    trainer.autoencoder.decoder.to("cpu")

    # Truncate to exact count
    all_gen_texts = all_gen_texts[:num_texts]
    all_trg_texts = all_trg_texts[:num_texts]
    all_pred_embeddings = torch.cat(all_pred_embeddings, dim=0)[:num_texts]

    # Compute rounding error (round-trip through decoder and re-encoder)
    trainer.autoencoder.encoder.to(device)
    trainer.autoencoder.decoder.to(device)
    rounding = compute_rounding_error(all_pred_embeddings, trainer.autoencoder, device)
    trainer.autoencoder.encoder.to("cpu")
    trainer.autoencoder.decoder.to("cpu")

    # Compute text quality metrics
    from estimation.metrics import compute_metric
    from estimation.util import truncate_text

    max_seq_len = trainer.cfg.dataset.max_sequence_len
    gen_truncated = truncate_text(all_gen_texts, max_seq_len, 1)
    trg_truncated = truncate_text(all_trg_texts, max_seq_len, 1)

    metrics = {}
    metrics.update(rounding)

    if len(gen_truncated) > 0 and len(trg_truncated) > 0:
        try:
            metrics["mauve"] = compute_metric("mauve",
                                               predictions=gen_truncated,
                                               references=trg_truncated)
        except Exception as e:
            print(f"MAUVE computation failed: {e}")
            metrics["mauve"] = None

    if len(gen_truncated) > 0:
        try:
            metrics["ppl"] = compute_metric("ppl",
                                             predictions=gen_truncated,
                                             references=None)
        except Exception as e:
            print(f"PPL computation failed: {e}")
            metrics["ppl"] = None

        try:
            metrics["diversity"] = compute_metric("div",
                                                   predictions=gen_truncated,
                                                   references=None)
        except Exception as e:
            print(f"Diversity computation failed: {e}")
            metrics["diversity"] = None

    metrics["num_generated"] = len(all_gen_texts)
    metrics["num_non_empty"] = len(gen_truncated)

    # Sample generated texts
    metrics["sample_texts"] = all_gen_texts[:10]

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE with Cosmos pipeline")
    parser.add_argument("--dataset", type=str, default="rocstories")
    parser.add_argument("--autoencoder_ckpt", type=str, required=True)
    parser.add_argument("--diffusion_ckpt", type=str, required=True)
    parser.add_argument("--sae_ckpt", type=str, default=None,
                        help="Path to SAE checkpoint")
    parser.add_argument("--dense_ckpt", type=str, default=None,
                        help="Path to Dense AE checkpoint (ablation)")
    parser.add_argument("--sae_type", type=str, default="topk")
    parser.add_argument("--expansion_factor", type=int, default=4)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--num_texts", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_latents", type=int, default=16)
    parser.add_argument("--max_position_embeddings", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build Hydra config programmatically
    from utils.hydra_utils import load_config
    project_root = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(
        project_root,
        os.path.join(project_root, "conf"),
        overrides=[f"dataset={args.dataset}"],
    )

    # Override config values
    cfg.ddp.enabled = False
    cfg.ddp.local_rank = 0
    cfg.ddp.global_rank = 0
    cfg.training = ""
    cfg.encoder.latent.num_latents = args.num_latents
    cfg.encoder.embedding.max_position_embeddings = args.max_position_embeddings
    cfg.decoder.latent.num_latents = args.num_latents
    cfg.decoder.embedding.max_position_embeddings = args.max_position_embeddings
    cfg.autoencoder.model.load_checkpoint = args.autoencoder_ckpt
    cfg.diffusion.model.load_checkpoint = args.diffusion_ckpt
    cfg.diffusion.training.batch_size = args.batch_size
    cfg.diffusion.training.batch_size_per_gpu = args.batch_size
    cfg.diffusion.generation.num_gen_texts = args.num_texts
    cfg.project.checkpoint_dir = args.checkpoint_dir

    # Load trainer
    from diffusion_trainer import DiffusionTrainer
    trainer = DiffusionTrainer(cfg)
    trainer.restore_checkpoint()

    results = {}

    # 1. Baseline (no SAE)
    print("\n=== Evaluating BASELINE (no SAE) ===")
    results["baseline"] = evaluate_model(
        trainer, sae=None, use_sae=False,
        num_texts=args.num_texts, device=device
    )
    print_metrics("Baseline", results["baseline"])

    # Read latent dim from config
    d_input = cfg.encoder.latent.dim

    # 2. SAE-filtered
    if args.sae_ckpt:
        print(f"\n=== Evaluating SAE ({args.sae_type}, {args.expansion_factor}x, k={args.k}) ===")
        sae = TopKSparseAutoencoder(
            d_input=d_input, expansion_factor=args.expansion_factor, k=args.k
        ) if args.sae_type == "topk" else DenseAutoencoder(
            d_input=d_input, expansion_factor=args.expansion_factor
        )
        state = torch.load(args.sae_ckpt, map_location="cpu")
        sae.load_state_dict(state["sae_state_dict"] if "sae_state_dict" in state else state)
        sae.to(device).eval()

        results["sae"] = evaluate_model(
            trainer, sae=sae, use_sae=True,
            num_texts=args.num_texts, device=device
        )
        print_metrics("SAE", results["sae"])

    # 3. Dense AE ablation
    if args.dense_ckpt:
        print(f"\n=== Evaluating Dense AE ({args.expansion_factor}x) ===")
        dense_ae = DenseAutoencoder(d_input=d_input, expansion_factor=args.expansion_factor)
        state = torch.load(args.dense_ckpt, map_location="cpu")
        dense_ae.load_state_dict(state["sae_state_dict"] if "sae_state_dict" in state else state)
        dense_ae.to(device).eval()

        results["dense_ae"] = evaluate_model(
            trainer, sae=dense_ae, use_sae=True,
            num_texts=args.num_texts, device=device
        )
        print_metrics("Dense AE", results["dense_ae"])

    # Save results
    # Remove non-serializable items
    save_results = {}
    for model_name, metrics in results.items():
        save_results[model_name] = {
            k: v for k, v in metrics.items()
            if k != "sample_texts" or isinstance(v, (list, str))
        }

    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # Print comparison table
    print_comparison_table(results)


def print_metrics(name: str, metrics: dict):
    """Print metrics for a single model variant."""
    print(f"\n{name}:")
    for key in ["mauve", "diversity", "ppl", "rounding_error_mean",
                 "rounding_error_median", "num_generated", "num_non_empty"]:
        val = metrics.get(key)
        if val is not None:
            if isinstance(val, float):
                print(f"  {key}: {val:.5f}")
            else:
                print(f"  {key}: {val}")


def print_comparison_table(results: dict):
    """Print side-by-side comparison table."""
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    header = f"{'Metric':<25}"
    for model in results:
        header += f"{model:>15}"
    print(header)
    print("-" * 80)

    metrics_to_show = [
        ("MAUVE", "mauve", True),
        ("Diversity", "diversity", True),
        ("PPL", "ppl", False),
        ("Rounding Error", "rounding_error_mean", False),
        ("Rounding Error (med)", "rounding_error_median", False),
    ]

    for label, key, _higher_better in metrics_to_show:
        row = f"{label:<25}"
        for model in results:
            val = results[model].get(key)
            if val is not None and isinstance(val, (int, float)):
                row += f"{val:>15.5f}"
            else:
                row += f"{'N/A':>15}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    main()
