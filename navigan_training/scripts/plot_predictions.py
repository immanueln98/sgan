#!/usr/bin/env python3
"""Render predicted trajectories vs ground truth for sample scenes.

For each sampled scene in the split, plots all agents' observed past (solid),
ground-truth future (dotted), and model prediction (dashed). A d_safe ring is
drawn around every agent at the first prediction step so pedestrian-clearance
violations are visible at a glance.

Usage:
    python scripts/plot_predictions.py runs/zara1_scratch/navigan_checkpoint.pt \
        --dataset zara1 --split val --num-scenes 6
    python scripts/plot_predictions.py runs/zara1_safe/navigan_checkpoint.pt \
        --variant best_safety --d_safe 1.0 -o plots/
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from navigan_training.data.loader import data_loader
from navigan_training.models import LateAttentionFullGenerator, TrajectoryDiscriminator
from navigan_training.utils import (AttrDict, discriminator_ctor_kwargs,
                                    get_dset_path, load_checkpoint,
                                    model_ctor_kwargs, relative_to_abs)

VARIANTS = ('final', 'best_ade', 'best_fde', 'best_safety')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--dataset', default=None)
    p.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    p.add_argument('--datasets-root', default='datasets')
    p.add_argument('--variant', default='final', choices=VARIANTS)
    p.add_argument('--num-scenes', type=int, default=6,
                   help='How many scenes to render (one subplot each)')
    p.add_argument('--d_safe', type=float, default=None,
                   help='Override d_safe ring radius')
    p.add_argument('--batch-size', type=int, default=1,
                   help='Loader batch size; 1 = one scene per batch')
    p.add_argument('-o', '--output-dir', default=None)
    p.add_argument('--dpi', type=int, default=120)
    p.add_argument('--device', default=None)
    return p.parse_args()


def _select_state(ck, variant):
    if variant == 'final':
        return ck['g_state'], ck['d_state']
    g = ck.get(f'g_{variant}_state'); d = ck.get(f'd_{variant}_state')
    if g is None or d is None:
        raise SystemExit(f'Checkpoint missing "{variant}" snapshot. Try --variant final.')
    return g, d


def _plot_scene(ax, obs, gt, pred, d_safe, title):
    """obs: (obs_len, N, 2), gt/pred: (pred_len, N, 2). Agent 0 highlighted."""
    n_agents = obs.shape[1]
    for i in range(n_agents):
        is_robot = (i == 0)
        color = 'tab:red' if is_robot else f'C{(i + 1) % 10}'
        zorder = 3 if is_robot else 2
        ax.plot(obs[:, i, 0], obs[:, i, 1], '-', color=color, linewidth=1.6,
                zorder=zorder, label='robot obs' if is_robot else None)
        ax.plot(gt[:, i, 0], gt[:, i, 1], ':', color=color, linewidth=1.2,
                zorder=zorder, label='robot gt' if is_robot else None)
        ax.plot(pred[:, i, 0], pred[:, i, 1], '--', color=color, linewidth=1.4,
                zorder=zorder, label='robot pred' if is_robot else None)
        # Marker at obs->pred boundary (current pose)
        ax.plot(obs[-1, i, 0], obs[-1, i, 1], 'o', color=color, markersize=5,
                zorder=zorder + 1)
        # d_safe ring around peds at start of prediction
        if not is_robot:
            ax.add_patch(plt.Circle((obs[-1, i, 0], obs[-1, i, 1]), d_safe,
                                    fill=False, edgecolor=color,
                                    linestyle='-', linewidth=0.6, alpha=0.4))
    ax.set_title(title, fontsize=9)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)


def main():
    args = parse_args()
    device = torch.device(args.device
                          or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ck = load_checkpoint(args.checkpoint)
    ck_args = AttrDict(ck['args']) if isinstance(ck['args'], dict) \
        else AttrDict(vars(ck['args']))

    dataset = args.dataset or ck_args.get('dataset_name')
    d_safe = args.d_safe if args.d_safe is not None else ck_args.get('d_safe', 0.5)

    g = LateAttentionFullGenerator(**model_ctor_kwargs(ck_args))
    d = TrajectoryDiscriminator(**discriminator_ctor_kwargs(ck_args))
    g_state, d_state = _select_state(ck, args.variant)
    g.load_state_dict(g_state); d.load_state_dict(d_state)
    g = g.to(device).eval(); d = d.to(device).eval()

    path = get_dset_path(args.datasets_root, dataset, args.split)
    cfg = dict(ck_args); cfg['batch_size'] = args.batch_size
    cfg.setdefault('loader_num_workers', 0)
    _, loader, _ = data_loader(cfg, path, distributed=False, shuffle=False)

    n = args.num_scenes
    cols = min(3, n); rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.5 * rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    legend_ax = None
    plotted = 0

    with torch.no_grad():
        for batch in loader:
            if plotted >= n:
                break
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _nl,
             _lm, seq_start_end, _goals, goals_rel) = [t.to(device) for t in batch]
            pred_rel, _aux = g(obs_traj, obs_traj_rel, seq_start_end,
                               goal_input=goals_rel.squeeze(0),
                               seq_len=cfg['pred_len'],
                               goal_aggr=cfg.get('goal_aggr', 0.5))
            pred_abs = relative_to_abs(pred_rel, obs_traj[0])

            # Walk per-sequence (each scene is one (start, end) slice)
            for s, e in seq_start_end.cpu().numpy():
                if plotted >= n:
                    break
                ax = axes_flat[plotted]
                _plot_scene(ax,
                            obs_traj[:, s:e, :].cpu().numpy(),
                            pred_traj_gt[:, s:e, :].cpu().numpy(),
                            pred_abs[:, s:e, :].cpu().numpy(),
                            d_safe,
                            f'scene {plotted}  (N={e - s} agents)')
                if legend_ax is None:
                    legend_ax = ax
                plotted += 1

    # Hide unused axes
    for k in range(plotted, len(axes_flat)):
        axes_flat[k].set_visible(False)

    if legend_ax is not None:
        # Add line-style legend explaining solid/dotted/dashed (color-agnostic)
        handles = [
            plt.Line2D([], [], color='k', linestyle='-',  label='observed past'),
            plt.Line2D([], [], color='k', linestyle=':',  label='ground-truth future'),
            plt.Line2D([], [], color='k', linestyle='--', label='predicted future'),
            plt.Line2D([], [], color='tab:red', linestyle='-', linewidth=2,
                       label='robot (agent 0)'),
            plt.Line2D([], [], color='grey', marker='o', linestyle='',
                       label=f'd_safe={d_safe} m ring'),
        ]
        fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
                   bbox_to_anchor=(0.5, -0.02))

    stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    fig.suptitle(f'{stem} — {dataset}/{args.split}  (variant={args.variant}, '
                 f'd_safe={d_safe})')
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir,
                       f'{stem}_predictions_{dataset}_{args.split}_{args.variant}.png')
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    print(f'[plot_predictions] wrote {out}  ({plotted} scenes)', file=sys.stderr)


if __name__ == '__main__':
    main()
