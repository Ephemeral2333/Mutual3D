from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .scannetv2_inst import ScanNetDataset

__all__ = ['ScanNetDataset', 'build_dataset', 'build_dataloader']


def build_dataset(data_cfg, logger):
    assert 'type' in data_cfg
    _data_cfg = data_cfg.copy()
    _data_cfg['logger'] = logger
    data_type = _data_cfg.pop('type')
    if data_type == 'scannetv2':
        return ScanNetDataset(**_data_cfg)
    else:
        raise ValueError(f'Unknown dataset type {data_type!r}')


def build_dataloader(dataset,
                     batch_size: int = 1,
                     num_workers: int = 1,
                     training: bool = True,
                     dist: bool = False,
                     persistent_workers: bool = True):
    shuffle = training
    sampler = DistributedSampler(dataset, shuffle=shuffle) if dist else None
    if sampler is not None:
        shuffle = False

    if training:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            persistent_workers=persistent_workers)
    else:
        assert batch_size == 1, 'Evaluation must use batch_size == 1'
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn,
            shuffle=False,
            sampler=sampler,
            drop_last=False,
            pin_memory=True,
            persistent_workers=persistent_workers)
