import torch
from omegaconf import DictConfig


def prepare_encodings_noising(encodings_shape: torch.Tensor, delta: float):
    noise = torch.randn(encodings_shape)
    sigma = (1 - delta ** 2) ** 0.5
    return delta, noise * sigma


def prepare_encodings_masking(encodings_shape: torch.Tensor, attention_mask: torch.Tensor, mlm_probability: float):
    # Create a mask for the encoder latents
    probability_matrix = torch.full(encodings_shape[:2], mlm_probability)
    probability_matrix[~attention_mask] = 0. # no mask for padding tokens 
    mask = torch.bernoulli(probability_matrix)
    mask = mask.bool()
    
    return mask


def apply_corruption(encodings: torch.Tensor, mask: torch.Tensor, alpha: torch.Tensor, noise: torch.Tensor):
    corrupted_encodings = encodings.clone()
    corrupted_encodings[mask] = 0.
    corrupted_encodings = encodings * alpha + noise
    return corrupted_encodings


def prepare_corruption(encodings_shape: tuple, attention_mask: torch.Tensor, config: DictConfig):
    """
    Prepare the masking for the encoder latents.
    corrupted_encodings = encodings.clone()
    corrupted_encodings[mask] = 0.
    corrupted_encodings = encodings * alpha + noise
    mask: 
        [
            [0, 1, 0, 0, 0], # masking
            [0, 1, 1, 0, 0], # masking
            [1, 1, 0, 1, 1], # masking
            [0, 0, 0, 0, 0], # noising
            [0, 0, 0, 0, 0], # noising
            [0, 0, 0, 0, 0], # noising
        ]
    alpha:
        [
            [1, 1, 1, 1, 1], # masking
            [1, 1, 1, 1, 1], # masking
            [1, 1, 1, 1, 1], # masking
            [a, a, a, a, a], # noising
            [a, a, a, a, a], # noising
            [a, a, a, a, a], # noising
        ]
    noise:
        [
            [0, 0, 0, 0, 0], # masking
            [0, 0, 0, 0, 0], # masking
            [0, 0, 0, 0, 0], # masking
            [s * e, s * e, s * e, s * e, s * e], # noising
            [s * e, s * e, s * e, s * e, s * e], # noising
            [s * e, s * e, s * e, s * e, s * e], # noising
        ]
    """
    num_samples_to_mask = int(config.masking.weight * encodings_shape[0])
    num_samples_to_noise = int(config.gaussian_noise.weight * encodings_shape[0])

    mask = torch.zeros((encodings_shape[0], encodings_shape[1])).bool()
    alpha = torch.ones(encodings_shape)
    noise = torch.zeros(encodings_shape)

    # encodings masking
    if num_samples_to_mask > 0:
        mask[:num_samples_to_mask] = prepare_encodings_masking(
            (num_samples_to_mask, encodings_shape[1], encodings_shape[2]), 
            attention_mask[:num_samples_to_mask], 
            config.masking.encodings_mlm_probability
        )

    # noise masking
    if num_samples_to_noise > 0:
        alpha[num_samples_to_mask:num_samples_to_mask + num_samples_to_noise], noise[num_samples_to_mask:num_samples_to_mask + num_samples_to_noise] = prepare_encodings_noising(
            (num_samples_to_noise, encodings_shape[1], encodings_shape[2]), 
            config.gaussian_noise.delta
        )

    # Apply the mask to the encoder latents
    corrupted_attenion_mask = attention_mask & ~mask

    return corrupted_attenion_mask, mask, alpha, noise