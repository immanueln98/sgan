"""DataLoader factory. Optional DistributedSampler for DDP."""
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .trajectories import TrajectoryDataset, seq_collate


def data_loader(cfg, path, distributed=False, shuffle=True):
    """Build (dataset, loader) for a split dir.

    cfg fields used: obs_len, pred_len, skip, delim, batch_size, loader_num_workers.
    """
    dset = TrajectoryDataset(
        path,
        obs_len=cfg['obs_len'],
        pred_len=cfg['pred_len'],
        skip=cfg['skip'],
        delim=cfg.get('delim', 'tab'),
    )

    sampler = DistributedSampler(dset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dset,
        batch_size=cfg['batch_size'],
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=cfg.get('loader_num_workers', 4),
        collate_fn=seq_collate,
        pin_memory=True,
        drop_last=distributed,
    )
    return dset, loader, sampler
