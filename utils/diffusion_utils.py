import os
import torch
import random
import numpy as np
from copy import deepcopy
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.functional import cross_entropy


def masked_mean(tensor, mask):
    return torch.sum(tensor * mask[:, :, None], dim=[0, 1]) / torch.sum(mask)


def masked_std(tensor, mask):
    mean = masked_mean(tensor, mask)
    return torch.sqrt(torch.sum(tensor ** 2 * mask[:, :, None], dim=[0, 1]) / torch.sum(mask) - mean ** 2)


def mse_loss(inputs, targets, mask=None):
    if mask is None:
        mask = torch.ones(
            (targets.shape[0], targets.shape[1]),
            device=inputs.device,
            requires_grad=False,
            dtype=torch.int64,
        )
    losses = torch.mean(torch.square(inputs - targets), dim=-1)
    losses = losses * mask
    loss = torch.sum(losses) / torch.sum(mask)
    return loss


def get_stat(z, mask=None):
    if mask is None:
        mask = torch.ones(
            (z.shape[0], z.shape[1]),
            device=z.device,
            requires_grad=False,
            dtype=torch.int64,
        )

    mean = masked_mean(z, mask)
    std = masked_std(z, mask)
    stat_dict = {
        "mean": torch.mean(mean),
        "std": torch.mean(std),
    }
    return stat_dict
