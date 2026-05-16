#!/usr/bin/env python3
"""Standalone evaluator for a trained NaviGAN checkpoint.

Loads a .pt, rebuilds the generator+discriminator from checkpoint['args']
(so architecture always matches the saved state_dict), and runs the same
evaluate() used during training on the requested dataset split.

Supports picking which state_dict to evaluate via --variant:
    final        — last saved state (default)
    best_ade     — lowest val ADE seen during training
    best_fde     — lowest val FDE seen during training
    best_safety  — lowest val resist_count seen during training

Examples:
    # Eval production ckpt on zara1 val (used to verify resist_loss reconstruction)
    python scripts/eval_checkpoint.py models/benchmark_zara1_with_model.pt \
        --dataset zara1 --split val

    # Compare deploy options after a safety-tuned run
    python scripts/eval_checkpoint.py runs/zara1_safe/navigan_checkpoint.pt \
        --dataset zara1 --split val --variant best_safety
    python scripts/eval_checkpoint.py runs/zara1_safe/navigan_checkpoint.pt \
        --dataset zara1 --split val --variant best_ade

    # Final test-split numbers for the paper / report
    python scripts/eval_checkpoint.py runs/zara1_scratch/navigan_checkpoint.pt \
        --dataset zara1 --split test --variant best_ade
"""
import argparse
import json
import logging
import sys

import torch

from navigan_training.data.loader import data_loader
from navigan_training.models import LateAttentionFullGenerator, TrajectoryDiscriminator
from navigan_training.train_loop import evaluate
from navigan_training.utils import (AttrDict, discriminator_ctor_kwargs,
                                    get_dset_path, load_checkpoint,
                                    model_ctor_kwargs)

VARIANTS = ('final', 'best_ade', 'best_fde', 'best_safety')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint', help='Path to .pt')
    p.add_argument('--dataset', default=None,
                   help='Dataset name (default: from checkpoint args)')
    p.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    p.add_argument('--datasets-root', default='datasets')
    p.add_argument('--variant', default='final', choices=VARIANTS)
    p.add_argument('--d_safe', type=float, default=None,
                   help='Override d_safe used for resist_loss/count')
    p.add_argument('--limit', type=int, default=None,
                   help='Cap number of trajectories scored (debug)')
    p.add_argument('--json', action='store_true',
                   help='Emit metrics as a single JSON line on stdout')
    p.add_argument('--device', default=None, help='cuda|cpu (auto)')
    return p.parse_args()


def _select_state(checkpoint, variant):
    """Return (g_state, d_state, label_for_log). final is always present;
    best_* slots may be None if training hadn't logged val metrics yet."""
    if variant == 'final':
        return checkpoint['g_state'], checkpoint['d_state'], 'final'

    g_key = f'g_{variant}_state'
    d_key = f'd_{variant}_state'
    g_state = checkpoint.get(g_key)
    d_state = checkpoint.get(d_key)
    if g_state is None or d_state is None:
        raise SystemExit(
            f'Checkpoint has no "{variant}" snapshot — keys {g_key}/{d_key} are missing or '
            f'None. Either the run never logged a val eval, or this is an older checkpoint '
            f'format that predates multi-criterion best tracking. Try --variant final.'
        )
    meta = checkpoint.get('bests', {}).get(variant, {})
    label = f'{variant} (t={meta.get("t", "?")}, {meta.get("metric")}={meta.get("value")})'
    return g_state, d_state, label


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        stream=sys.stdout)
    log = logging.getLogger('eval')

    device = torch.device(args.device
                          or ('cuda' if torch.cuda.is_available() else 'cpu'))
    log.info(f'Loading checkpoint: {args.checkpoint}')
    ck = load_checkpoint(args.checkpoint)

    ck_args = AttrDict(ck['args']) if isinstance(ck['args'], dict) \
        else AttrDict(vars(ck['args']))

    dataset = args.dataset or ck_args.get('dataset_name')
    if not dataset:
        raise SystemExit('No dataset given and checkpoint args lack dataset_name.')
    d_safe = args.d_safe if args.d_safe is not None else ck_args.get('d_safe', 0.5)
    log.info(f'Dataset: {dataset} / {args.split}, d_safe={d_safe}, variant={args.variant}')

    g = LateAttentionFullGenerator(**model_ctor_kwargs(ck_args))
    d = TrajectoryDiscriminator(**discriminator_ctor_kwargs(ck_args))
    g_state, d_state, label = _select_state(ck, args.variant)
    g.load_state_dict(g_state)
    d.load_state_dict(d_state)
    g, d = g.to(device).eval(), d.to(device).eval()
    log.info(f'Loaded weights: {label}')

    path = get_dset_path(args.datasets_root, dataset, args.split)
    log.info(f'Path: {path}')
    cfg = dict(ck_args)
    cfg['d_safe'] = d_safe
    cfg['num_samples_check'] = args.limit or (1 << 30)
    cfg.setdefault('loader_num_workers', 4)
    cfg.setdefault('batch_size', 32)
    _, loader, _ = data_loader(cfg, path, distributed=False, shuffle=False)

    metrics = evaluate(cfg, loader, g, d, device, limit=bool(args.limit))
    if args.json:
        print(json.dumps({
            'checkpoint': args.checkpoint, 'variant': args.variant,
            'dataset': dataset, 'split': args.split, 'd_safe': d_safe,
            **{k: float(v) for k, v in metrics.items()},
        }))
    else:
        log.info('Results:')
        for k in ('ade', 'fde', 'resist_loss', 'resist_count', 'd_loss'):
            log.info(f'  {k:14s}: {metrics[k]:.4f}')


if __name__ == '__main__':
    main()
