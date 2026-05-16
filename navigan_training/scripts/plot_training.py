#!/usr/bin/env python3
"""Plot training curves from a navigan_training checkpoint.

Renders:
  - Generator loss components (l2, adv, resist, intention, total) vs iter
  - Discriminator loss vs iter
  - Val + train ADE vs iter
  - Val + train FDE vs iter
  - Val + train resist_loss vs iter
  - Val + train resist_count vs iter

Vertical lines mark the best_ade, best_fde, best_safety snapshots.

Usage:
    python scripts/plot_training.py runs/zara1_scratch/navigan_checkpoint.pt
    python scripts/plot_training.py runs/zara1_safe/navigan_checkpoint.pt -o plots/
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # No display needed on HPC / headless
import matplotlib.pyplot as plt

from navigan_training.utils import load_checkpoint

BEST_COLORS = {
    'best_ade':    'tab:green',
    'best_fde':    'tab:olive',
    'best_safety': 'tab:red',
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('-o', '--output-dir', default=None,
                   help='Output dir (default: alongside the checkpoint)')
    p.add_argument('--no-bests', action='store_true',
                   help='Skip vertical lines for best-iter markers')
    p.add_argument('--dpi', type=int, default=120)
    return p.parse_args()


def _mark_bests(ax, checkpoint, skip=False):
    if skip:
        return
    bests = checkpoint.get('bests', {})
    for slot, meta in bests.items():
        t = meta.get('t')
        if t is None:
            continue
        ax.axvline(t, color=BEST_COLORS.get(slot, 'grey'),
                   linestyle=':', linewidth=1, alpha=0.7, label=slot)


def _plot_g_losses(ax, ck):
    ts = ck.get('losses_ts', [])
    if not ts:
        ax.text(0.5, 0.5, 'no G loss data', ha='center', va='center',
                transform=ax.transAxes); return
    for key, series in sorted(ck.get('G_losses', {}).items()):
        if key.endswith('_count'):  # plotted separately
            continue
        n = min(len(ts), len(series))
        if n > 0:
            ax.plot(ts[:n], series[:n], label=key.replace('G_', ''), linewidth=1)
    ax.set_title('Generator losses'); ax.set_xlabel('iter'); ax.set_ylabel('loss')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


def _plot_d_loss(ax, ck):
    ts = ck.get('losses_ts', [])
    series = ck.get('D_losses', {}).get('D_total_loss', [])
    n = min(len(ts), len(series))
    if n > 0:
        ax.plot(ts[:n], series[:n], color='tab:purple', linewidth=1)
    ax.set_title('Discriminator loss'); ax.set_xlabel('iter'); ax.set_ylabel('loss')
    ax.grid(alpha=0.3)


def _plot_eval_metric(ax, ck, key, ylabel, mark_bests):
    ts = ck.get('sample_ts', [])
    val = ck.get('metrics_val', {}).get(key, [])
    tr = ck.get('metrics_train', {}).get(key, [])
    nv = min(len(ts), len(val)); nt = min(len(ts), len(tr))
    if nv > 0:
        ax.plot(ts[:nv], val[:nv], label='val', color='tab:blue', linewidth=1.2)
    if nt > 0:
        ax.plot(ts[:nt], tr[:nt], label='train', color='tab:orange',
                linewidth=1, alpha=0.7)
    _mark_bests(ax, ck, skip=not mark_bests)
    ax.set_title(key); ax.set_xlabel('iter'); ax.set_ylabel(ylabel)
    ax.legend(fontsize=7); ax.grid(alpha=0.3)


def main():
    args = parse_args()
    ck = load_checkpoint(args.checkpoint)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    _plot_g_losses(axes[0, 0], ck)
    _plot_d_loss(axes[0, 1], ck)
    _plot_eval_metric(axes[0, 2], ck, 'ade',          'metres',    not args.no_bests)
    _plot_eval_metric(axes[1, 0], ck, 'fde',          'metres',    not args.no_bests)
    _plot_eval_metric(axes[1, 1], ck, 'resist_loss',  'hinge',     not args.no_bests)
    _plot_eval_metric(axes[1, 2], ck, 'resist_count', '#viol/batch', not args.no_bests)

    ck_args = ck.get('args', {})
    dataset = ck_args.get('dataset_name', '?') if isinstance(ck_args, dict) \
        else getattr(ck_args, 'dataset_name', '?')
    d_safe = ck_args.get('d_safe', '?') if isinstance(ck_args, dict) \
        else getattr(ck_args, 'd_safe', '?')
    fig.suptitle(f'{stem}  ({dataset}, d_safe={d_safe})')
    fig.tight_layout()

    out_path = os.path.join(out_dir, f'{stem}_training.png')
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
    print(f'[plot_training] wrote {out_path}', file=sys.stderr)

    bests = ck.get('bests', {})
    if bests:
        print('[plot_training] best snapshots:', file=sys.stderr)
        for slot, meta in bests.items():
            print(f'  {slot:12s}  t={meta.get("t")}  '
                  f'{meta.get("metric")}={meta.get("value")}', file=sys.stderr)


if __name__ == '__main__':
    main()
