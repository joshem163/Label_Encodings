import logging

import numpy as np
import torch
from sklearn.model_selection import ShuffleSplit
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import index2mask, set_dataset_attr

RANDOM_SPLIT_RATIOS = [0.6, 0.2, 0.2]


def prepare_splits(dataset):
    """Always generate a random 60/20/20 train/val/test split."""
    logging.info(
        f"[Split] dataset={cfg.dataset.name} mode={cfg.dataset.split_mode} "
        f"policy=random_fixed_60_20_20"
    )
    setup_random_split(dataset)


def setup_random_split(dataset):
    """Generate random 60/20/20 splits using cfg.seed."""
    split_ratios = RANDOM_SPLIT_RATIOS
    if list(getattr(cfg.dataset, 'split', [])) != split_ratios:
        logging.info(
            f"[Split] overriding cfg.dataset.split={getattr(cfg.dataset, 'split', None)} "
            f"-> {split_ratios}"
        )
    logging.info(f"[Split] using=random ratio={split_ratios} seed={cfg.seed}")

    # If a labeled_mask exists (e.g., datasets with unlabeled nodes),
    # restrict splitting to labeled nodes only.
    if hasattr(dataset.data, 'labeled_mask'):
        candidate_indices = np.where(dataset.data.labeled_mask.cpu().numpy())[0]
    else:
        candidate_indices = np.arange(dataset.data.y.shape[0])

    train_rel, val_test_rel = next(
        ShuffleSplit(train_size=split_ratios[0], random_state=cfg.seed).split(
            np.zeros(len(candidate_indices))
        )
    )
    train_index = candidate_indices[train_rel]
    val_test_index = candidate_indices[val_test_rel]

    val_test_ratio = split_ratios[1] / (1 - split_ratios[0])
    val_rel, test_rel = next(
        ShuffleSplit(train_size=val_test_ratio, random_state=cfg.seed).split(
            np.zeros(len(val_test_index))
        )
    )
    val_index = val_test_index[val_rel]
    test_index = val_test_index[test_rel]

    set_dataset_splits(dataset, [train_index, val_index, test_index])


def set_dataset_splits(dataset, splits):
    """Set given splits to the dataset object."""
    split_sets = []
    split_sizes = []
    for split_idx in splits:
        if isinstance(split_idx, torch.Tensor):
            vals = split_idx.detach().cpu().view(-1).tolist()
        elif isinstance(split_idx, np.ndarray):
            vals = split_idx.reshape(-1).tolist()
        else:
            vals = list(split_idx)
        split_sets.append(set(int(v) for v in vals))
        split_sizes.append(len(vals))

    for i in range(len(split_sets) - 1):
        for j in range(i + 1, len(split_sets)):
            n_intersect = len(split_sets[i] & split_sets[j])
            if n_intersect != 0:
                raise ValueError(
                    f"Splits must not have intersecting indices: "
                    f"split #{i} (n = {split_sizes[i]}) and "
                    f"split #{j} (n = {split_sizes[j]}) have "
                    f"{n_intersect} intersecting indices"
                )

    task_level = cfg.dataset.task
    if task_level == 'node':
        split_names = ['train_mask', 'val_mask', 'test_mask']
        for split_name, split_index in zip(split_names, splits):
            if not isinstance(split_index, torch.Tensor):
                split_index = torch.as_tensor(split_index, dtype=torch.long)
            mask = index2mask(split_index, size=dataset.data.y.shape[0])
            set_dataset_attr(dataset, split_name, mask, len(mask))

    elif task_level == 'graph':
        split_names = ['train_graph_index', 'val_graph_index', 'test_graph_index']
        for split_name, split_index in zip(split_names, splits):
            set_dataset_attr(dataset, split_name, split_index, len(split_index))

    else:
        raise ValueError(f"Unsupported dataset task level: {task_level}")
