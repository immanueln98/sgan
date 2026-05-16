#!/usr/bin/env python3
"""Training entry point. Single-GPU OR torch.distributed multi-GPU.

Usage (single GPU):
    python train.py --config configs/zara1_scratch.yaml

Usage (multi-GPU via torchrun):
    torchrun --nproc_per_node=4 train.py --distributed --config configs/zara1_scratch.yaml

Resume:
    python train.py --config configs/zara1_resume.yaml

Override fields ad-hoc:
    python train.py --config configs/zara1_safe.yaml --max-iter 200 --d_safe 1.0
"""
import argparse
import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from navigan_training.data.loader import data_loader
from navigan_training.models import LateAttentionFullGenerator, TrajectoryDiscriminator
from navigan_training.train_loop import build_fresh_checkpoint, run_training
from navigan_training.utils import (AttrDict, discriminator_ctor_kwargs,
                                    get_dset_path, load_checkpoint,
                                    model_ctor_kwargs, seed_everything)

LOG_FORMAT = '[%(levelname)s %(asctime)s %(name)s] %(message)s'
logger = logging.getLogger('train')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--distributed', action='store_true',
                   help='Init DDP via torchrun env vars')
    p.add_argument('--datasets-root', default='datasets',
                   help='Root dir containing dataset_name/{train,val,test}/ subdirs')
    p.add_argument('--max-iter', type=int, default=None,
                   help='Override num_iterations (smoke tests)')
    p.add_argument('--d_safe', type=float, default=None,
                   help='Override d_safe (metres). Tunes safety margin.')
    p.add_argument('--resist-weight', type=float, default=None,
                   help='Override resist_loss_weight')
    return p.parse_args()


def init_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)


def load_cfg(path, overrides):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if overrides.max_iter is not None:
        cfg['num_iterations'] = overrides.max_iter
    if overrides.d_safe is not None:
        cfg['d_safe'] = overrides.d_safe
    if overrides.resist_weight is not None:
        cfg['resist_loss_weight'] = overrides.resist_weight
    return cfg


def main():
    args = parse_args()

    if args.distributed:
        local_rank, rank, world_size = init_distributed()
        is_master = (rank == 0)
        device = torch.device(f'cuda:{local_rank}')
    else:
        is_master = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rank, world_size = 0, 1

    logging.basicConfig(level=logging.INFO if is_master else logging.WARNING,
                        format=LOG_FORMAT, stream=sys.stdout)

    cfg = load_cfg(args.config, args)
    seed_everything(cfg.get('seed', 42) + rank)
    if is_master:
        logger.info(f'World size: {world_size}, device: {device}')
        logger.info(f'Loaded config: {args.config}')

    train_path = get_dset_path(args.datasets_root, cfg['dataset_name'], 'train')
    val_path = get_dset_path(args.datasets_root, cfg['dataset_name'], 'val')
    if is_master:
        logger.info(f'train: {train_path}\nval:   {val_path}')

    train_dset, train_loader, train_sampler = data_loader(
        cfg, train_path, distributed=args.distributed, shuffle=True)
    _, val_loader, _ = data_loader(
        cfg, val_path, distributed=False, shuffle=False)

    iters_per_epoch = max(1, len(train_dset) // cfg['batch_size'] // cfg['d_steps'])
    if cfg.get('num_epochs'):
        cfg['num_iterations'] = int(iters_per_epoch * cfg['num_epochs'])
    if args.max_iter is not None:
        cfg['num_iterations'] = args.max_iter
    if is_master:
        logger.info(f'iters_per_epoch={iters_per_epoch}, total iters={cfg["num_iterations"]}')

    # Resume vs fresh
    resume_from = cfg.get('resume_from')
    if resume_from and os.path.isfile(resume_from):
        if is_master:
            logger.info(f'Resuming from {resume_from}')
        checkpoint = load_checkpoint(resume_from)
        ckpt_args = AttrDict(checkpoint['args']) if isinstance(checkpoint['args'], dict) \
            else AttrDict(vars(checkpoint['args']))
        gen_kwargs = model_ctor_kwargs(ckpt_args)
        disc_kwargs = discriminator_ctor_kwargs(ckpt_args)
        generator = LateAttentionFullGenerator(**gen_kwargs)
        discriminator = TrajectoryDiscriminator(**disc_kwargs)
        generator.load_state_dict(checkpoint['g_state'])
        discriminator.load_state_dict(checkpoint['d_state'])
        start_t = checkpoint.get('counters', {}).get('t', 0) or 0
        start_epoch = checkpoint.get('counters', {}).get('epoch', 0) or 0
        checkpoint.setdefault('restore_ts', []).append(start_t)
    else:
        if is_master:
            logger.info('Training from scratch')
        gen_kwargs = model_ctor_kwargs(cfg)
        disc_kwargs = discriminator_ctor_kwargs(cfg)
        generator = LateAttentionFullGenerator(**gen_kwargs)
        discriminator = TrajectoryDiscriminator(**disc_kwargs)
        generator.apply(init_weights)
        discriminator.apply(init_weights)
        checkpoint = build_fresh_checkpoint(cfg)
        start_t, start_epoch = 0, 0

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    optimizer_g = optim.Adam(generator.parameters(), lr=cfg['g_learning_rate'])
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg['d_learning_rate'])
    if resume_from and os.path.isfile(resume_from):
        if checkpoint.get('g_optim_state'):
            optimizer_g.load_state_dict(checkpoint['g_optim_state'])
        if checkpoint.get('d_optim_state'):
            optimizer_d.load_state_dict(checkpoint['d_optim_state'])

    if args.distributed:
        generator = DDP(generator, device_ids=[local_rank], find_unused_parameters=True)
        discriminator = DDP(discriminator, device_ids=[local_rank])

    out_dir = cfg.get('output_dir', 'runs/run')
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'{cfg.get("checkpoint_name", "navigan_checkpoint")}.pt')

    generator.train()
    discriminator.train()

    try:
        run_training(cfg, generator, discriminator, optimizer_g, optimizer_d,
                     train_loader, val_loader, train_sampler, device,
                     checkpoint, start_t, start_epoch, output_path, is_master=is_master)
    finally:
        if args.distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
