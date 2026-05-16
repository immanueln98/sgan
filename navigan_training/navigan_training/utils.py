"""Utilities: AttrDict, dset path resolution, seeding, ckpt save/load helpers."""
import os
import random
from pathlib import Path

import numpy as np
import torch


class AttrDict(dict):
    """Dict with attribute access. Used so checkpoint['args'] -> AttrDict can be
    passed as **kwargs to model ctors and also accessed as args.foo."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dset_path(datasets_root: str, dset_name: str, dset_type: str) -> str:
    """Resolve datasets/{dset_name}/{dset_type}/ — populated by download_data.sh."""
    return os.path.join(datasets_root, dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """rel_traj: (seq_len, batch, 2). start_pos: (batch, 2). Output: (seq_len, batch, 2).

    Per the dataset convention, rel positions are relative to the FIRST observation
    point (obs_traj[0]) — not the last. Pass obs_traj[0] as start_pos.
    """
    rel_traj = rel_traj.permute(1, 0, 2)
    start_pos = start_pos.unsqueeze(1)
    return (rel_traj + start_pos).permute(1, 0, 2)


def get_total_norm(parameters, norm_type=2):
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(norm_type)
        total += param_norm.item() ** norm_type
    return total ** (1.0 / norm_type)


def model_ctor_kwargs(args):
    """Filter args dict down to LateAttentionFullGenerator __init__ kwargs.

    Checkpoint args dict has training hyperparams mixed in with model kwargs.
    Use the model's __init__ signature to pick only what matters.
    """
    from .models.late_attention import LateAttentionFullGenerator
    valid = set(LateAttentionFullGenerator.__init__.__code__.co_varnames)
    # Map sgan naming -> model param naming
    aliases = {
        'encoder_h_dim_g': 'encoder_h_dim',
        'decoder_h_dim_g': 'decoder_h_dim',
    }
    out = {}
    for k, v in args.items():
        k = aliases.get(k, k)
        if k in valid and k != 'self':
            out[k] = v
    return out


def discriminator_ctor_kwargs(args):
    from .models.late_attention import TrajectoryDiscriminator
    valid = set(TrajectoryDiscriminator.__init__.__code__.co_varnames)
    aliases = {'encoder_h_dim_d': 'h_dim'}
    out = {}
    for k, v in args.items():
        k = aliases.get(k, k)
        if k in valid and k != 'self':
            out[k] = v
    return out


def save_checkpoint(checkpoint: dict, path: str):
    """Atomic-ish save: write to .tmp then rename."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = path + '.tmp'
    torch.save(checkpoint, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
