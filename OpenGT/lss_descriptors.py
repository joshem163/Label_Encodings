import math
from collections import deque
from typing import Optional, Tuple

import torch


def build_adjacency_list(edge_index: torch.Tensor, num_nodes: int, undirected: bool = True):
    adj = [[] for _ in range(num_nodes)]
    row, col = edge_index.cpu()

    for u, v in zip(row.tolist(), col.tolist()):
        adj[u].append(v)
        if undirected and u != v:
            adj[v].append(u)

    return adj


def exact_k_hop_annuli(adj, source: int, max_k: int = 3):
    annuli = {k: [] for k in range(max_k + 1)}

    visited = {source}
    q = deque([(source, 0)])

    while q:
        node, dist = q.popleft()

        if dist > max_k:
            continue

        annuli[dist].append(node)

        if dist == max_k:
            continue

        for nbr in adj[node]:
            if nbr not in visited:
                visited.add(nbr)
                q.append((nbr, dist + 1))

    return annuli


def hide_labels_by_index(
    y: torch.Tensor,
    hide_idx,
    unknown_label: int = -1,
) -> torch.Tensor:
    y_masked = y.clone()

    if not torch.is_tensor(hide_idx):
        hide_idx = torch.tensor(hide_idx, dtype=torch.long, device=y.device)
    else:
        hide_idx = hide_idx.to(y.device).long()

    y_masked[hide_idx] = unknown_label
    return y_masked


def compute_annulus_descriptor_all_nodes(
    data,
    y_masked: torch.Tensor,
    num_classes: Optional[int] = None,
    max_k: int = 3,
    unknown_label: int = -1,
    undirected: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """CPU reference implementation."""
    edge_index = data.edge_index.cpu()
    num_nodes = data.num_nodes
    y_masked = y_masked.cpu().view(-1)

    known_mask = (y_masked != unknown_label)

    if num_classes is None:
        known_y = y_masked[known_mask]
        if known_y.numel() == 0:
            raise ValueError("No known labels available to infer num_classes.")
        num_classes = int(known_y.max().item()) + 1

    adj = build_adjacency_list(edge_index, num_nodes, undirected=undirected)

    block_dim = num_classes + 2
    out_dim = max_k * block_dim
    desc = torch.empty((num_nodes, out_dim), dtype=dtype)

    uniform = torch.full((num_classes,), 1.0 / num_classes, dtype=dtype)

    for v in range(num_nodes):
        annuli = exact_k_hop_annuli(adj, source=v, max_k=max_k)
        blocks = []

        for k in range(1, max_k + 1):
            Ak = annuli[k]
            nk = len(Ak)

            if nk == 0:
                pk = uniform.clone()
                cov_k = torch.tensor([0.0], dtype=dtype)
                ell_k = torch.tensor([0.0], dtype=dtype)
            else:
                Ak_tensor = torch.tensor(Ak, dtype=torch.long)
                known_in_annulus = known_mask[Ak_tensor]
                Aklab = Ak_tensor[known_in_annulus]
                mk = Aklab.numel()

                if mk == 0:
                    pk = uniform.clone()
                else:
                    cls = y_masked[Aklab].long()
                    counts = torch.bincount(cls, minlength=num_classes).to(dtype)
                    pk = counts / mk

                cov_k = torch.tensor([mk / (nk + 1.0)], dtype=dtype)
                ell_k = torch.tensor([math.log(mk + 1.0)], dtype=dtype)

            block = torch.cat([pk, cov_k, ell_k], dim=0)
            blocks.append(block)

        desc[v] = torch.cat(blocks, dim=0)

    return desc


def compute_annulus_descriptor_all_nodes_gpu(
    data,
    y_masked: torch.Tensor,
    num_classes: Optional[int] = None,
    max_k: int = 3,
    unknown_label: int = -1,
    undirected: bool = True,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 256,
) -> torch.Tensor:
    """Batched GPU implementation using sparse adjacency and frontier expansion."""
    if not torch.cuda.is_available():
        raise RuntimeError("GPU backend requested but CUDA is not available.")

    device = data.edge_index.device
    if device.type != "cuda":
        # Keep all ops on CUDA for speed.
        device = torch.device("cuda")

    edge_index = data.edge_index.to(device)
    num_nodes = int(getattr(data, "num_nodes", data.x.size(0)))
    y_masked = y_masked.to(device).view(-1)

    known_mask = (y_masked != unknown_label)

    if num_classes is None:
        known_y = y_masked[known_mask]
        if known_y.numel() == 0:
            raise ValueError("No known labels available to infer num_classes.")
        num_classes = int(known_y.max().item()) + 1

    row, col = edge_index[0], edge_index[1]
    if undirected:
        row = torch.cat([row, col], dim=0)
        col = torch.cat([col, row[:edge_index.size(1)]], dim=0)
    values = torch.ones(row.size(0), device=device, dtype=dtype)
    adj = torch.sparse_coo_tensor(
        torch.stack([row, col], dim=0), values, (num_nodes, num_nodes), device=device
    ).coalesce()
    adj_t = adj.transpose(0, 1).coalesce()

    cls = y_masked.clone().long()
    cls[~known_mask] = 0
    known_onehot = torch.nn.functional.one_hot(cls, num_classes=num_classes).to(dtype)
    known_onehot = known_onehot * known_mask.to(dtype).unsqueeze(1)

    block_dim = num_classes + 2
    out_dim = max_k * block_dim
    desc = torch.empty((num_nodes, out_dim), dtype=dtype, device=device)
    uniform = torch.full((num_classes,), 1.0 / num_classes, dtype=dtype, device=device)

    source_all = torch.arange(num_nodes, device=device)
    bs = max(1, int(batch_size))
    for start in range(0, num_nodes, bs):
        end = min(start + bs, num_nodes)
        src = source_all[start:end]
        bsz = src.numel()

        frontier = torch.zeros((bsz, num_nodes), dtype=torch.bool, device=device)
        frontier[torch.arange(bsz, device=device), src] = True
        visited = frontier.clone()

        for k in range(1, max_k + 1):
            nbr_scores = torch.sparse.mm(adj_t, frontier.to(dtype).t()).t()
            next_frontier = (nbr_scores > 0) & (~visited)
            visited |= next_frontier

            nk = next_frontier.sum(dim=1).to(dtype)
            known_frontier = next_frontier & known_mask.unsqueeze(0)
            mk = known_frontier.sum(dim=1).to(dtype)

            counts = known_frontier.to(dtype) @ known_onehot
            pk = counts / mk.unsqueeze(1).clamp_min(1.0)
            no_known = (mk == 0)
            if no_known.any():
                pk[no_known] = uniform

            cov_k = mk / (nk + 1.0)
            ell_k = torch.log(mk + 1.0)

            offset = (k - 1) * block_dim
            desc[start:end, offset:offset + num_classes] = pk
            desc[start:end, offset + num_classes] = cov_k
            desc[start:end, offset + num_classes + 1] = ell_k

            frontier = next_frontier

    return desc


def build_descriptor_from_split_indices(
    data,
    test_idx,
    val_idx=None,
    hide_test_only: bool = True,
    max_k: int = 3,
    unknown_label: int = -1,
    num_classes: Optional[int] = None,
    undirected: bool = True,
    dtype: torch.dtype = torch.float32,
    backend: str = "auto",
    gpu_batch_size: int = 256,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_tensor(test_idx):
        test_idx = torch.tensor(test_idx, dtype=torch.long)
    else:
        test_idx = test_idx.long()

    if hide_test_only:
        hide_idx = test_idx
    else:
        if val_idx is None:
            raise ValueError("val_idx must be provided when hide_test_only=False.")
        if not torch.is_tensor(val_idx):
            val_idx = torch.tensor(val_idx, dtype=torch.long)
        else:
            val_idx = val_idx.long()
        hide_idx = torch.cat([val_idx, test_idx], dim=0)

    y_masked = hide_labels_by_index(
        y=data.y.view(-1),
        hide_idx=hide_idx,
        unknown_label=unknown_label,
    )

    backend_norm = backend.lower()
    if backend_norm == "auto":
        use_gpu = bool(torch.cuda.is_available() and data.edge_index.device.type == "cuda")
    elif backend_norm == "gpu":
        use_gpu = True
    elif backend_norm == "cpu":
        use_gpu = False
    else:
        raise ValueError(f"Invalid backend '{backend}'. Use one of: auto|cpu|gpu")

    if use_gpu:
        desc = compute_annulus_descriptor_all_nodes_gpu(
            data=data,
            y_masked=y_masked,
            num_classes=num_classes,
            max_k=max_k,
            unknown_label=unknown_label,
            undirected=undirected,
            dtype=dtype,
            batch_size=gpu_batch_size,
        )
    else:
        desc = compute_annulus_descriptor_all_nodes(
            data=data,
            y_masked=y_masked,
            num_classes=num_classes,
            max_k=max_k,
            unknown_label=unknown_label,
            undirected=undirected,
            dtype=dtype,
        )

    return desc, y_masked
