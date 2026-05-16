#!/usr/bin/env python3
"""Inspect a NaviGAN/SGAN .pt checkpoint without instantiating any model.

Prints:
  - Top-level keys in the saved dict
  - Full `args` dict (constructor + training hyperparams used to make this ckpt)
  - Critical config that LateAttentionFullGenerator needs (noise_dim, pooling,
    goal_dim, spatial_dim, etc.) — call out if missing
  - Training counters (iteration `t`, `epoch`, best iterations)
  - Each state_dict: entry count, total params, sample tensor keys
  - Loss/metric history: lengths + first/last values per series

Run on a machine with torch installed (Jetson, HPC node, or any env with
`pip install torch`). Output is human-readable text — copy-paste back so we can
build the training script against the actual saved args.

Usage:
    python inspect_checkpoint.py path/to/benchmark_zara1_with_model.pt
"""

import argparse
import pprint
import sys

import torch


def main(path: str) -> None:
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location='cpu')

    print('=' * 72)
    print('TOP-LEVEL KEYS')
    print('=' * 72)
    for k in ckpt.keys():
        v = ckpt[k]
        if isinstance(v, dict):
            print(f'  {k:<30s} dict   ({len(v):>5d} entries)')
        elif isinstance(v, list):
            print(f'  {k:<30s} list   ({len(v):>5d} entries)')
        elif v is None:
            print(f'  {k:<30s} None')
        else:
            print(f'  {k:<30s} {type(v).__name__:<8s} = {v}')

    print()
    print('=' * 72)
    print('ARGS (full)')
    print('=' * 72)
    args = ckpt.get('args')
    if hasattr(args, 'keys') and not isinstance(args, dict):
        args = dict(args)
    pprint.pprint(args, sort_dicts=True, width=100)

    print()
    print('=' * 72)
    print('CRITICAL CONFIG FOR LateAttentionFullGenerator')
    print('=' * 72)
    crit = [
        'obs_len', 'pred_len',
        'embedding_dim', 'encoder_h_dim_g', 'decoder_h_dim_g',
        'mlp_dim', 'num_layers',
        'noise_dim', 'noise_type', 'noise_mix_type',
        'pooling_type', 'pool_every_timestep',
        'bottleneck_dim', 'neighborhood_size', 'grid_size',
        'goal_dim', 'spatial_dim',
        'batch_norm', 'dropout',
        'd_type', 'encoder_h_dim_d',
        'best_k', 'l2_loss_weight',
        'g_learning_rate', 'd_learning_rate', 'd_steps', 'g_steps',
        'clipping_threshold_g', 'clipping_threshold_d',
        'batch_size', 'num_iterations', 'num_epochs',
        'dataset_name', 'skip', 'delim',
    ]
    if isinstance(args, dict):
        for k in crit:
            v = args.get(k, '<<NOT SET>>')
            print(f'  {k:<28s} = {v}')
    else:
        print('  args is not a dict — cannot extract')

    print()
    print('=' * 72)
    print('COUNTERS / BEST CHECKPOINT MARKERS')
    print('=' * 72)
    pprint.pprint(ckpt.get('counters'))
    for k in ('best_t', 'best_t_nl', 'restore_ts'):
        print(f'  {k:<28s} = {ckpt.get(k)}')

    print()
    print('=' * 72)
    print('STATE_DICTS — presence + param counts')
    print('=' * 72)
    state_keys = [
        'g_state', 'd_state',
        'g_optim_state', 'd_optim_state',
        'g_best_state', 'd_best_state',
        'g_best_nl_state', 'd_best_state_nl',
        'g_waypointbest_state',
    ]
    for sk in state_keys:
        v = ckpt.get(sk)
        if v is None:
            print(f'  {sk:<28s} <<MISSING>>')
            continue
        if isinstance(v, dict):
            tensor_keys = [k for k, t in v.items() if hasattr(t, 'shape')]
            total_params = sum(
                t.numel() for t in v.values() if hasattr(t, 'numel')
            )
            print(f'  {sk:<28s} {len(v):>4d} entries   '
                  f'{total_params:>12,} params')
            if tensor_keys:
                print(f'      first 5: {tensor_keys[:5]}')
                print(f'      last  5: {tensor_keys[-5:]}')
        else:
            print(f'  {sk:<28s} {v}')

    print()
    print('=' * 72)
    print('LOSS / METRIC HISTORY (sample first + last per series)')
    print('=' * 72)
    for hist_key in ('G_losses', 'D_losses', 'metrics_val', 'metrics_train'):
        h = ckpt.get(hist_key)
        if not h:
            continue
        print(f'  {hist_key}:')
        for k, lst in h.items():
            if not lst:
                continue
            try:
                print(f'    {k:<25s} len={len(lst):<5d} '
                      f'first={float(lst[0]):.4f}  last={float(lst[-1]):.4f}')
            except (TypeError, ValueError):
                print(f'    {k:<25s} len={len(lst):<5d} (non-numeric)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to .pt checkpoint')
    args = parser.parse_args()
    sys.exit(main(args.checkpoint))
