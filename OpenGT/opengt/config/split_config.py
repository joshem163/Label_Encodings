from torch_geometric.graphgym.register import register_config


@register_config('split')
def set_cfg_split(cfg):
    """Reconfigure the default config value for dataset split options.

    Returns:
        Reconfigured split configuration use by the experiment.
    """

    # Default to random 60/20/20 train/val/test split
    cfg.dataset.split_mode = 'random'

    # Default split ratios: 60% train, 20% val, 20% test
    cfg.dataset.split = [0.6, 0.2, 0.2]

    # Choose a particular split to use if multiple splits are available
    cfg.dataset.split_index = 0

    # Dir to cache cross-validation splits
    cfg.dataset.split_dir = './splits'

    # Choose to run multiple splits in one program execution, if set,
    # takes the precedence over cfg.dataset.split_index for split selection
    cfg.run_multiple_splits = []
