import logging
import os.path as osp
import time
import math
from functools import partial

import numpy as np
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import (Actor, Planetoid, Amazon, Coauthor,
                                      WebKB, WikipediaNetwork, WikiCS, CoraFull,
                                      LINKXDataset, HeterophilousGraphDataset,
                                      AttributedGraphDataset)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loader import load_pyg, load_ogb, set_dataset_attr
from torch_geometric.graphgym.register import register_loader

from opengt.loader.dataset.critical import Critical
from opengt.loader.split_generator import (prepare_splits,
                                             set_dataset_splits)
from opengt.transform.posenc_stats import compute_posenc_stats
from opengt.transform.transforms import (pre_transform_in_memory,
                                           typecast_x, concat_x_and_pos,
                                           clip_graphs_to_size, move_node_feat_to_x)
from opengt.transform.expander_edges import generate_random_expander
from opengt.transform.dist_transforms import (add_dist_features, add_reverse_edges,
                                                 add_self_loops, effective_resistances,
                                                 effective_resistance_embedding,
                                                 effective_resistances_from_embedding)
from opengt.transform.multihop_prep import generate_multihop_adj


dataset_drive_url = {
    'snap-patents': '1ldh23TSY1PwXia6dU0MYcpyEgX-w3Hia',
    'pokec': '1dNs5E7BrWJbgcHeQ_zuy5Ozp2tRCWG0y',
    'yelp-chi': '1fAXtTVQS4CfEk4asqrFw9EPmlUPGbGtJ',
}

_DATASET_CACHE = {}


def _manual_is_undirected(data):
    """Backend-agnostic undirected check that avoids pyg::index_sort."""
    edge_index = getattr(data, 'edge_index', None)
    if edge_index is None or edge_index.numel() == 0:
        return True
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    edges = set(zip(src, dst))
    return all((v, u) in edges for (u, v) in edges)


def _safe_is_undirected(data):
    """Try fast path first, then fallback when backend ops are missing."""
    try:
        return data.is_undirected()
    except Exception as e:
        logging.warning(f"Falling back to manual undirected check: {e}")
        return _manual_is_undirected(data)


def _cache_signature(format, name, dataset_dir):
    """Build a cache key that is stable across seeds/split indices."""
    pe_sig = []
    for key, pecfg in cfg.items():
        if str(key).startswith('posenc_'):
            enabled = bool(getattr(pecfg, 'enable', False))
            dim_pe = getattr(pecfg, 'dim_pe', None)
            times_func = None
            if hasattr(pecfg, 'kernel'):
                times_func = getattr(pecfg.kernel, 'times_func', None)
            max_freqs = None
            if hasattr(pecfg, 'eigen'):
                max_freqs = getattr(pecfg.eigen, 'max_freqs', None)
            pe_sig.append((str(key), enabled, dim_pe, str(times_func), max_freqs))
    pe_sig = tuple(sorted(pe_sig))

    prep_sig = (
        bool(cfg.prep.exp), int(cfg.prep.exp_count), int(cfg.prep.exp_deg),
        str(cfg.prep.exp_algorithm), int(cfg.prep.exp_max_num_iters),
        bool(cfg.prep.dist_enable), int(cfg.prep.dist_cutoff),
        int(cfg.prep.rb_order),
        int(cfg.metis.patches), bool(cfg.metis.enable),
        float(cfg.metis.drop_rate), int(cfg.metis.num_hops),
        int(cfg.metis.patch_rw_dim), int(cfg.metis.patch_num_diff),
    )
    return (format, name, osp.abspath(dataset_dir), pe_sig, prep_sig)


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    # logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if isinstance(dataset.data.y, list):
            # A special case for ogbg-code2 dataset.
            logging.info(f"  num classes: n/a")
        elif dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    cache_key = _cache_signature(format, name, dataset_dir)
    if cache_key in _DATASET_CACHE:
        dataset = _DATASET_CACHE[cache_key]
        logging.info(f"[*] Reusing dataset cache for '{name}' from '{format}'.")
    elif format.startswith('PyG-'):
        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'Actor':
            if name != 'none':
                raise ValueError(f"Actor class provides only one dataset.")
            dataset = Actor(dataset_dir)

        elif pyg_dataset_id == 'Amazon':
            dataset = Amazon(dataset_dir, name)

        elif pyg_dataset_id == 'Coauthor':
            dataset = Coauthor(dataset_dir, name)

        elif pyg_dataset_id == 'WikiCS':
            dataset = WikiCS(dataset_dir)

        elif pyg_dataset_id == 'CoraFull':
            dataset = CoraFull(dataset_dir)

        
        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'WebKB':
            dataset = WebKB(dataset_dir, name)

        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError(f"crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'LINKXDataset':
            dataset = LINKXDataset(dataset_dir, name)
            # Some LINKXDataset graphs (e.g. Penn94) have -1 labels for
            # unlabeled nodes. Mark those as excluded from all splits by
            # zeroing their label and recording a mask so split_generator
            # will never assign them to train/val/test.
            data = dataset._data
            if (data.y < 0).any():
                unlabeled = (data.y < 0)
                data.y[unlabeled] = 0  # dummy value, won't appear in any split
                data.labeled_mask = ~unlabeled  # used by split_generator below
            else:
                data.labeled_mask = torch.ones(data.num_nodes, dtype=torch.bool)

        elif pyg_dataset_id == 'HeterophilousGraphDataset':
            dataset = HeterophilousGraphDataset(dataset_dir, name)

        elif pyg_dataset_id == 'AttributedGraphDataset':
            dataset = AttributedGraphDataset(dataset_dir, name)
            # Flickr stores features as a sparse float64 matrix; convert to
            # a dense float32 tensor so downstream code can use it normally.
            data = dataset._data
            if data.x is not None and data.x.is_sparse:
                data.x = data.x.to_dense().to(torch.float32)
            elif data.x is not None and data.x.dtype == torch.float64:
                data.x = data.x.to(torch.float32)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'OGB':
        if name.startswith('ogbn'):
            dataset = preformat_ogbn(dataset_dir, name)
        else:
            raise ValueError(f"Unsupported OGB dataset: {name}")
    elif format == 'Critical':
        dataset_dir = osp.join(dataset_dir, 'Critical')
        dataset = Critical(dataset_dir, name)
        
    else:
        raise ValueError(f"Unknown data format: {format}")
    if cache_key not in _DATASET_CACHE:
        log_loaded_dataset(dataset, format, name)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('posenc_') and pecfg.enable and (not key.startswith('posenc_ER')):
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} PE kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list and cache_key not in _DATASET_CACHE:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(_safe_is_undirected(d) for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # Other preprocessings:
    # adding expander edges:
    if cfg.prep.exp and cache_key not in _DATASET_CACHE:
        for j in range(cfg.prep.exp_count):
            start = time.perf_counter()
            logging.info(f"Adding expander edges (round {j}) ...")
            pre_transform_in_memory(dataset,
                                    partial(generate_random_expander,
                                            degree = cfg.prep.exp_deg,
                                            algorithm = cfg.prep.exp_algorithm,
                                            rng = None,
                                            max_num_iters = cfg.prep.exp_max_num_iters,
                                            exp_index = j),
                                    show_progress=True
                                    )
            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")


    # adding shortest path features
    if cfg.prep.dist_enable and cache_key not in _DATASET_CACHE:
        start = time.perf_counter()
        logging.info(f"Precalculating node distances and shortest paths ...")
        is_undirected = _safe_is_undirected(dataset[0])
        Max_N = max([data.num_nodes for data in dataset])
        pre_transform_in_memory(dataset,
                                partial(add_dist_features,
                                        max_n = Max_N,
                                        is_undirected = is_undirected,
                                        cutoff = cfg.prep.dist_cutoff),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    if cfg.prep.rb_order > 1 and cache_key not in _DATASET_CACHE:
        start = time.perf_counter()
        logging.info(f"Generating multi-hop adjacency matrices ...")
        pre_transform_in_memory(dataset,
                                partial(generate_multihop_adj,
                                        cfg = cfg),
                                show_progress=True
                                )
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")


    # adding effective resistance features
    if (cfg.posenc_ERN.enable or cfg.posenc_ERE.enable) and cache_key not in _DATASET_CACHE:
        start = time.perf_counter()
        logging.info(f"Precalculating effective resistance for graphs ...")
        
        MaxK = max(
            [
                min(
                math.ceil(data.num_nodes//2), 
                math.ceil(8 * math.log(data.num_edges) / (cfg.posenc_ERN.accuracy**2))
                ) 
                for data in dataset
            ]
            )

        cfg.posenc_ERN.er_dim = MaxK
        logging.info(f"Choosing ER pos enc dim = {MaxK}")

        pre_transform_in_memory(dataset,
                                partial(effective_resistance_embedding,
                                        MaxK = MaxK,
                                        accuracy = cfg.posenc_ERN.accuracy,
                                        which_method = 0),
                                show_progress=True
                                )

        pre_transform_in_memory(dataset,
                        partial(effective_resistances_from_embedding,
                        normalize_per_node = False),
                        show_progress=True
                        )

        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    # graph partition transform

    if cfg.metis.patches > 0 and cache_key not in _DATASET_CACHE:
        try:
            from opengt.transform.graph_partition import GraphPartitionTransform
        except Exception as e:
            logging.warning(
                f"Skipping graph partition transform: GraphPartitionTransform unavailable ({e})"
            )
        else:
            start = time.perf_counter()
            logging.info(f"Precomputing graph partition transform ...")

            pre_transform_in_memory(dataset, GraphPartitionTransform(n_patches=cfg.metis.patches,
                                                                     metis=cfg.metis.enable,
                                                                     drop_rate=cfg.metis.drop_rate,
                                                                     num_hops=cfg.metis.num_hops,
                                                                     is_directed=False,
                                                                     patch_rw_dim=cfg.metis.patch_rw_dim,
                                                                     patch_num_diff=cfg.metis.patch_num_diff),
                                    show_progress=True)

            elapsed = time.perf_counter() - start
            timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                      + f'{elapsed:.2f}'[-3:]
            logging.info(f"Done! Took {timestr}")
    
    
    dataset.data['extra_loss'] = torch.Tensor([0.0])
    if cache_key not in _DATASET_CACHE:
        _DATASET_CACHE[cache_key] = dataset

    # This could not be done earlier because the training wants 'train_mask' etc.
    # Now after using gnn.head: inductive_node this is ok.
    if name == 'ogbn-arxiv' or name == 'ogbn-proteins':
      return dataset
    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute in-degree histogram if needed for PNAConv.
    if cfg.gt.layer_type.startswith('PNAConv') and len(cfg.gt.pna_degrees) == 0:
        cfg.gt.pna_degrees = compute_indegree_histogram(
            dataset[dataset.data['train_graph_index']])

    return dataset


def compute_indegree_histogram(dataset):
    """Compute histogram of in-degree of nodes needed for PNAConv.

    Args:
        dataset: PyG Dataset object

    Returns:
        List where i-th value is the number of nodes with in-degree equal to `i`
    """
    from torch_geometric.utils import degree

    deg = torch.zeros(1000, dtype=torch.long)
    max_degree = 0
    for data in dataset:
        d = degree(data.edge_index[1],
                   num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, d.max().item())
        deg += torch.bincount(d, minlength=deg.numel())
    return deg.numpy().tolist()[:max_degree + 1]


def preformat_ogbn(dataset_dir, name):
  if name == 'ogbn-arxiv' or name == 'ogbn-proteins':
    dataset = PygNodePropPredDataset(name=name)
    if name == 'ogbn-arxiv':
      pre_transform_in_memory(dataset, partial(add_reverse_edges))
      if cfg.prep.add_self_loops:
        pre_transform_in_memory(dataset, partial(add_self_loops))
    if name == 'ogbn-proteins':
      pre_transform_in_memory(dataset, partial(move_node_feat_to_x))
      pre_transform_in_memory(dataset, partial(typecast_x, type_str='float'))
    split_dict = dataset.get_idx_split()
    split_dict['val'] = split_dict.pop('valid')
    dataset.split_idx = split_dict
    return dataset


  else:
     raise ValueError(f"Unknown ogbn dataset '{name}'.")
