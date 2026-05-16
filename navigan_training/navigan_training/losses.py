"""Loss functions.

Ports from naviGAN_paper/sgan/losses.py:
  - bce_loss, gan_g_loss, gan_d_loss, l2_loss
  - displacement_error, final_displacement_error

Reconstructed losses (not in source repo despite being baked into checkpoint args):
  - resist_loss: hinge-form pairwise repulsion. The safety-distance lever.
  - intention_loss: direct L2 on intention-branch rollout pre-attention fusion.
"""
import random

import torch


def bce_loss(input, target):
    """Numerically stable BCE with logits."""
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def gan_g_loss(scores_fake):
    y_fake = torch.ones_like(scores_fake) * random.uniform(0.7, 1.2)
    return bce_loss(scores_fake, y_fake)


def gan_d_loss(scores_real, scores_fake):
    y_real = torch.ones_like(scores_real) * random.uniform(0.7, 1.2)
    y_fake = torch.zeros_like(scores_fake) * random.uniform(0, 0.3)
    return bce_loss(scores_real, y_real) + bce_loss(scores_fake, y_fake)


def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):
    """pred_traj: (seq_len, batch, 2). loss_mask: (batch, seq_len)."""
    loss = (loss_mask.unsqueeze(2)
            * (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2)
    if mode == 'sum':
        return loss.sum()
    if mode == 'average':
        return loss.sum() / torch.numel(loss_mask.data)
    if mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)
    raise ValueError(f'Unknown l2_loss mode: {mode}')


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2)) ** 2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if consider_ped is not None:
        loss = loss * consider_ped
    return loss.sum() if mode == 'sum' else loss


def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    loss = (pred_pos_gt - pred_pos) ** 2
    loss = torch.sqrt(loss.sum(dim=1))
    if consider_ped is not None:
        loss = loss * consider_ped
    return loss if mode == 'raw' else loss.sum()


def resist_loss(pred_traj_abs, seq_start_end, d_safe=0.5):
    """Hinge-form pairwise repulsion over the prediction horizon.

      loss = mean_{seq} mean_{t} mean_{(i,j) violators} max(0, d_safe - ||x_i - x_j||)

    Why mean over violators (vs sum): the value stays scale-invariant to crowd
    density so the loss magnitude tracks the per-pair severity rather than the
    headcount. Matches the magnitudes seen in the trained checkpoint history
    (val resist 0.52 -> 0.21; far below what a sum would produce on zara1 with
    thousands of pairs per batch).

    Returns:
      (loss_scalar, violator_count) — scalar tensor + int tensor (sum of #pairs
      with d < d_safe across all seqs/timesteps). violator_count matches the
      `resist_count` series in the checkpoint metrics.
    """
    seq_len = pred_traj_abs.size(0)
    total_loss = pred_traj_abs.new_zeros(())
    total_violators = 0
    contributing_seqs = 0

    for start, end in seq_start_end:
        start, end = start.item(), end.item()
        num_ped = end - start
        if num_ped < 2:
            continue
        # (seq_len, num_ped, 2) -> pairwise distances per timestep
        traj = pred_traj_abs[:, start:end, :]
        # diff[t, i, j, :] = x_i(t) - x_j(t)
        diff = traj.unsqueeze(2) - traj.unsqueeze(1)
        dist = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-9)
        # mask out self-pairs and upper triangle (keep i < j only)
        mask = torch.triu(torch.ones(num_ped, num_ped, device=traj.device), diagonal=1).bool()
        hinge = torch.clamp(d_safe - dist, min=0.0)
        hinge = hinge * mask.unsqueeze(0)  # broadcast over time

        violators = (hinge > 0).sum().item()
        total_violators += violators
        if violators > 0:
            # mean over the contributing pairs
            total_loss = total_loss + hinge.sum() / max(violators, 1)
            contributing_seqs += 1

    if contributing_seqs > 0:
        total_loss = total_loss / contributing_seqs
    return total_loss, total_violators


def intention_loss(intention_traj_rel, pred_traj_gt_rel, loss_mask, mode='average'):
    """L2 on the intention-branch rollout (before attention fusion).

    intention_traj_rel comes from LateAttentionFullGenerator's auxiliary output:
        _, [attention, intent, social] = generator(...)
        loss = intention_loss(intent, pred_traj_gt_rel, loss_mask)

    Direct supervision pushes the intention branch toward ground-truth motion
    independent of the social-force branch — matches the trained checkpoint's
    `intention_loss_weight=0.05` schedule.
    """
    return l2_loss(intention_traj_rel, pred_traj_gt_rel, loss_mask, mode=mode)
