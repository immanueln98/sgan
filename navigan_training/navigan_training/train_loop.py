"""Training loop: discriminator_step, generator_step, evaluate, run_training."""
import gc
import logging
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

from .losses import (gan_d_loss, gan_g_loss, l2_loss, displacement_error,
                     final_displacement_error, resist_loss, intention_loss)
from .utils import get_total_norm, relative_to_abs, save_checkpoint

logger = logging.getLogger(__name__)


def _unpack(batch, device):
    return [t.to(device, non_blocking=True) for t in batch]


def _goal_input(goals_rel):
    """goals_rel from collator: (1, batch, 2). LateAttentionFullGenerator's
    add_goal reshapes to (batch, 2) via add_goal -> get_noise(inject_goal)."""
    return goals_rel.squeeze(0)


def discriminator_step(cfg, batch, generator, discriminator, optimizer_d, device):
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _non_linear,
     _loss_mask, seq_start_end, _goals, goals_rel) = _unpack(batch, device)

    pred_traj_fake_rel, _aux = generator(
        obs_traj, obs_traj_rel, seq_start_end,
        goal_input=_goal_input(goals_rel),
        seq_len=cfg['pred_len'],
        goal_aggr=cfg.get('goal_aggr', 0.5),
    )
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)

    scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)
    scores_fake = discriminator(traj_fake.detach(), traj_fake_rel.detach(), seq_start_end)
    loss = gan_d_loss(scores_real, scores_fake)

    optimizer_d.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.get('clipping_threshold_d', 0) > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(), cfg['clipping_threshold_d'])
    optimizer_d.step()
    return {'D_total_loss': loss.item()}


def generator_step(cfg, batch, generator, discriminator, optimizer_g, device):
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _non_linear,
     loss_mask, seq_start_end, _goals, goals_rel) = _unpack(batch, device)

    loss_mask = loss_mask[:, cfg['obs_len']:]
    losses = {}
    total = obs_traj.new_zeros(())

    pred_traj_fake_rel, aux = generator(
        obs_traj, obs_traj_rel, seq_start_end,
        goal_input=_goal_input(goals_rel),
        seq_len=cfg['pred_len'],
        goal_aggr=cfg.get('goal_aggr', 0.5),
    )
    _attention, intent_traj_rel, _social = aux
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])

    # L2 reconstruction
    if cfg.get('l2_loss_weight', 1.0) > 0:
        l2 = cfg['l2_loss_weight'] * l2_loss(
            pred_traj_fake_rel, pred_traj_gt_rel, loss_mask, mode='average')
        total = total + l2
        losses['G_l2_loss'] = l2.item()

    # Adversarial
    if cfg.get('gan_loss_weight', 1.0) > 0:
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        g_adv = cfg.get('gan_loss_weight', 1.0) * gan_g_loss(scores_fake)
        total = total + g_adv
        losses['G_discriminator_loss'] = g_adv.item()

    # Safety repulsion (the tunable knob)
    if cfg.get('resist_loss_weight', 0.0) > 0:
        r_loss, n_viol = resist_loss(
            pred_traj_fake, seq_start_end, d_safe=cfg.get('d_safe', 0.5))
        r_loss = cfg['resist_loss_weight'] * r_loss
        total = total + r_loss
        losses['G_resist_loss'] = r_loss.item()
        losses['G_resist_count'] = float(n_viol)

    # Intention-branch direct supervision
    if cfg.get('intention_loss_weight', 0.0) > 0:
        i_loss = cfg['intention_loss_weight'] * intention_loss(
            intent_traj_rel, pred_traj_gt_rel, loss_mask, mode='average')
        total = total + i_loss
        losses['G_intention_loss'] = i_loss.item()

    losses['G_total_loss'] = total.item()
    optimizer_g.zero_grad(set_to_none=True)
    total.backward()
    if cfg.get('clipping_threshold_g', 0) > 0:
        nn.utils.clip_grad_norm_(generator.parameters(), cfg['clipping_threshold_g'])
    optimizer_g.step()
    return losses


@torch.no_grad()
def evaluate(cfg, loader, generator, discriminator, device, limit=False):
    generator.eval()
    discriminator.eval()
    ade_sum = fde_sum = 0.0
    resist_sum = 0.0
    resist_count_sum = 0
    d_loss_sum = 0.0
    total_traj = 0
    n_batches = 0
    n_samples_cap = cfg.get('num_samples_check', 5000)

    for batch in loader:
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, _non_linear,
         _loss_mask, seq_start_end, _goals, goals_rel) = _unpack(batch, device)
        pred_traj_fake_rel, _aux = generator(
            obs_traj, obs_traj_rel, seq_start_end,
            goal_input=_goal_input(goals_rel),
            seq_len=cfg['pred_len'],
            goal_aggr=cfg.get('goal_aggr', 0.5),
        )
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[0])

        ade_sum += displacement_error(pred_traj_fake, pred_traj_gt).item()
        fde_sum += final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1]).item()

        r_loss, n_viol = resist_loss(
            pred_traj_fake, seq_start_end, d_safe=cfg.get('d_safe', 0.5))
        resist_sum += r_loss.item()
        resist_count_sum += n_viol

        traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
        traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
        traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
        traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
        scores_real = discriminator(traj_real, traj_real_rel, seq_start_end)
        scores_fake = discriminator(traj_fake, traj_fake_rel, seq_start_end)
        d_loss_sum += gan_d_loss(scores_real, scores_fake).item()

        total_traj += pred_traj_gt.size(1)
        n_batches += 1
        if limit and total_traj >= n_samples_cap:
            break

    generator.train()
    discriminator.train()
    return {
        'ade': ade_sum / (total_traj * cfg['pred_len']),
        'fde': fde_sum / total_traj,
        'resist_loss': resist_sum / max(n_batches, 1),
        'resist_count': resist_count_sum / max(n_batches, 1),
        'd_loss': d_loss_sum / max(n_batches, 1),
    }


def run_training(cfg, generator, discriminator, optimizer_g, optimizer_d,
                 train_loader, val_loader, train_sampler, device,
                 checkpoint, start_t, start_epoch, output_path, is_master=True):
    """Main loop. Mirrors sgan/train.py alternating d_steps/g_steps."""
    t = start_t
    epoch = start_epoch
    max_iter = cfg['num_iterations']
    d_steps = cfg.get('d_steps', 2)
    g_steps = cfg.get('g_steps', 1)
    losses_d, losses_g = {}, {}

    while t < max_iter:
        gc.collect()
        d_left, g_left = d_steps, g_steps
        epoch += 1
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_master:
            logger.info(f'Starting epoch {epoch} (t={t}/{max_iter})')
        t_epoch = time.time()

        for batch in train_loader:
            if d_left > 0:
                losses_d = discriminator_step(cfg, batch, generator, discriminator,
                                              optimizer_d, device)
                if is_master:
                    checkpoint['norm_d'].append(get_total_norm(discriminator.parameters()))
                d_left -= 1
            elif g_left > 0:
                losses_g = generator_step(cfg, batch, generator, discriminator,
                                          optimizer_g, device)
                if is_master:
                    checkpoint['norm_g'].append(get_total_norm(generator.parameters()))
                g_left -= 1

            if d_left > 0 or g_left > 0:
                continue

            if is_master and t % cfg.get('print_every', 50) == 0:
                logger.info(f't = {t + 1} / {max_iter}')
                for k, v in sorted(losses_d.items()):
                    logger.info(f'  [D] {k}: {v:.3f}')
                    checkpoint['D_losses'][k].append(v)
                for k, v in sorted(losses_g.items()):
                    logger.info(f'  [G] {k}: {v:.3f}')
                    checkpoint['G_losses'][k].append(v)
                checkpoint['losses_ts'].append(t)

            if is_master and t > 0 and t % cfg.get('checkpoint_every', 500) == 0:
                checkpoint['counters'] = {'t': t, 'epoch': epoch}
                checkpoint['sample_ts'].append(t)

                logger.info('Eval on val ...')
                metrics_val = evaluate(cfg, val_loader, generator, discriminator, device)
                logger.info('Eval on train (limited) ...')
                metrics_train = evaluate(cfg, train_loader, generator, discriminator,
                                         device, limit=True)
                for k, v in sorted(metrics_val.items()):
                    logger.info(f'  [val] {k}: {v:.4f}')
                    checkpoint['metrics_val'][k].append(v)
                for k, v in sorted(metrics_train.items()):
                    logger.info(f'  [train] {k}: {v:.4f}')
                    checkpoint['metrics_train'][k].append(v)

                _update_bests(checkpoint, metrics_val, t, generator, discriminator)

                checkpoint['g_state'] = _unwrap(generator).state_dict()
                checkpoint['g_optim_state'] = optimizer_g.state_dict()
                checkpoint['d_state'] = _unwrap(discriminator).state_dict()
                checkpoint['d_optim_state'] = optimizer_d.state_dict()
                logger.info(f'Saving checkpoint to {output_path}')
                save_checkpoint(checkpoint, output_path)

            t += 1
            d_left, g_left = d_steps, g_steps
            if t >= max_iter:
                break

        if is_master:
            logger.info(f'Epoch {epoch} took {time.time() - t_epoch:.1f}s')


def _unwrap(model):
    """Unwrap DDP for state_dict so saved keys don't have `module.` prefix."""
    return model.module if hasattr(model, 'module') else model


# Best-checkpoint tracking is multi-criterion. GAN val metrics oscillate epoch-to-epoch
# and no single scalar captures both prediction quality AND social safety, so we save
# the best snapshot under each criterion independently and let the deployer pick.
BEST_CRITERIA = {
    'best_ade':    ('ade',          'min'),  # accuracy: avg displacement error
    'best_fde':    ('fde',          'min'),  # accuracy: final displacement error
    'best_safety': ('resist_count', 'min'),  # safety: pedestrian-proximity violations
}


def _update_bests(checkpoint, metrics_val, t, generator, discriminator):
    """Compare current val metrics against tracked bests, snapshot on improvement."""
    bests = checkpoint.setdefault('bests', {})
    for slot, (metric_key, direction) in BEST_CRITERIA.items():
        current = metrics_val[metric_key]
        prev = bests.get(slot, {}).get('value')
        improved = (prev is None
                    or (direction == 'min' and current < prev)
                    or (direction == 'max' and current > prev))
        if improved:
            logger.info(f'New {slot} ({metric_key}={current:.4f})')
            bests[slot] = {'value': current, 't': t, 'metric': metric_key,
                           'all_metrics': dict(metrics_val)}
            checkpoint[f'g_{slot}_state'] = _unwrap(generator).state_dict()
            checkpoint[f'd_{slot}_state'] = _unwrap(discriminator).state_dict()


def build_fresh_checkpoint(cfg):
    ck = {
        'args': dict(cfg),
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        'losses_ts': [],
        'metrics_val': defaultdict(list),
        'metrics_train': defaultdict(list),
        'sample_ts': [],
        'restore_ts': [],
        'norm_g': [],
        'norm_d': [],
        'counters': {'t': 0, 'epoch': 0},
        'g_state': None, 'g_optim_state': None,
        'd_state': None, 'd_optim_state': None,
        'bests': {},
    }
    for slot in BEST_CRITERIA:
        ck[f'g_{slot}_state'] = None
        ck[f'd_{slot}_state'] = None
    return ck
