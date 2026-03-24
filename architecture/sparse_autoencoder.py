"""
Sparse Autoencoder (SAE) module for Cosmos latent diffusion.

Implements TopK sparse autoencoder and Dense (ReLU) autoencoder baseline.
Operates per-latent-vector: input dim=768, not flattened across N latent positions.
"""

import torch
import torch.nn as nn


class TopKSparseAutoencoder(nn.Module):
    """
    TopK Sparse Autoencoder.

    Encoder: s = TopK(W_enc * z + b_enc)   -- sparse codes
    Decoder: z_hat = W_dec * s + b_dec      -- reconstruction

    Decoder columns are unit-normalized after each optimizer step
    via normalize_decoder_weights().
    """

    def __init__(self, d_input: int = 768, expansion_factor: int = 4, k: int = 64):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_input * expansion_factor
        self.k = k

        self.encoder = nn.Linear(d_input, self.d_hidden)
        self.decoder = nn.Linear(self.d_hidden, d_input, bias=True)

        # Initialize encoder weights with Kaiming uniform
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        # Initialize decoder as transpose of encoder (tied init)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
        nn.init.zeros_(self.decoder.bias)

        # Normalize decoder columns at init
        self.normalize_decoder_weights()

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (batch, d_input) or (batch, N, d_input).
               If 3D, reshapes internally to (batch*N, d_input).

        Returns:
            z_hat: reconstructed latents, same shape as z
            s: sparse codes, shape (..., d_hidden)
            info: dict with auxiliary info (topk_indices, pre_activations)
        """
        input_shape = z.shape
        if z.dim() == 3:
            z = z.reshape(-1, self.d_input)

        # Encode
        pre_acts = self.encoder(z)  # (batch, d_hidden)

        # TopK sparsification
        topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
        s = torch.zeros_like(pre_acts)
        s.scatter_(1, topk_idx, topk_vals)

        # Decode
        z_hat = self.decoder(s)

        info = {
            "topk_indices": topk_idx,
            "pre_activations": pre_acts,
        }

        if len(input_shape) == 3:
            z_hat = z_hat.reshape(input_shape)
            s = s.reshape(input_shape[0], input_shape[1], -1)

        return z_hat, s, info

    @torch.no_grad()
    def normalize_decoder_weights(self):
        """Unit-normalize decoder columns (feature directions)."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.decoder.weight.div_(norms)

    def compute_loss(self, z: torch.Tensor):
        """
        Compute reconstruction MSE loss. TopK provides implicit sparsity.

        Args:
            z: input latents

        Returns:
            loss: scalar MSE loss
            info: dict with loss components and stats
            s: sparse codes (for dead feature tracking without a second forward pass)
        """
        z_hat, s, fwd_info = self.forward(z)
        mse = torch.mean((z - z_hat) ** 2)

        # Fraction of variance unexplained
        with torch.no_grad():
            var_z = torch.var(z)
            fvu = mse / var_z.clamp(min=1e-8)
            l0 = (s != 0).float().sum(dim=-1).mean()
            dead_mask = (s != 0).any(dim=0).float()  # per-feature alive indicator
            dead_frac = 1.0 - dead_mask.mean()

        info = {
            "mse": mse.item(),
            "fvu": fvu.item(),
            "l0": l0.item(),
            "dead_frac": dead_frac.item(),
        }
        return mse, info, s


class DenseAutoencoder(nn.Module):
    """
    Dense autoencoder baseline (ReLU instead of TopK).
    Same architecture as TopKSparseAutoencoder for fair ablation.
    """

    def __init__(self, d_input: int = 768, expansion_factor: int = 4):
        super().__init__()
        self.d_input = d_input
        self.d_hidden = d_input * expansion_factor

        self.encoder = nn.Linear(d_input, self.d_hidden)
        self.decoder = nn.Linear(self.d_hidden, d_input, bias=True)

        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: (batch, d_input) or (batch, N, d_input)

        Returns:
            z_hat: reconstructed latents
            h: hidden activations
            info: empty dict (for API compatibility)
        """
        input_shape = z.shape
        if z.dim() == 3:
            z = z.reshape(-1, self.d_input)

        h = torch.relu(self.encoder(z))
        z_hat = self.decoder(h)

        if len(input_shape) == 3:
            z_hat = z_hat.reshape(input_shape)
            h = h.reshape(input_shape[0], input_shape[1], -1)

        return z_hat, h, {}

    def compute_loss(self, z: torch.Tensor):
        """Compute reconstruction MSE loss."""
        z_hat, h, _ = self.forward(z)
        mse = torch.mean((z - z_hat) ** 2)

        with torch.no_grad():
            var_z = torch.var(z)
            fvu = mse / var_z.clamp(min=1e-8)
            l0 = (h != 0).float().sum(dim=-1).mean()
            dead_mask = (h != 0).any(dim=0).float()
            dead_frac = 1.0 - dead_mask.mean()

        info = {
            "mse": mse.item(),
            "fvu": fvu.item(),
            "l0": l0.item(),
            "dead_frac": dead_frac.item(),
        }
        return mse, info, h
