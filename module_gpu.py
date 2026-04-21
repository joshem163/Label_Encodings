import math
from collections import deque
from typing import Optional, Tuple

import torch


def cosine_similarity_torch(A, B, eps=1e-12):
    """
    A: [N, F]
    B: [C, F]
    return: [N, C]
    """
    dot = A @ B.t()
    An = A.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    Bn = B.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
    return dot / (An * Bn.t())


def similarity_torch(A, B, threshold=0.0):
    """
    Binary intersection similarity:
      sim(i,j) = sum_k [A[i,k] & B[j,k]]

    A: [N, F]
    B: [C, F]
    return: [N, C]
    """
    A_bin = (A > threshold).to(torch.float32)
    B_bin = (B > threshold).to(torch.float32)
    return A_bin @ B_bin.t()


def _to_index_tensor(idx, device):
    if torch.is_tensor(idx):
        return idx.to(device=device, dtype=torch.long)
    return torch.tensor(idx, device=device, dtype=torch.long)


def _build_train_mask(num_nodes, test_idx, device):
    test_idx = _to_index_tensor(test_idx, device)
    train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    train_mask[test_idx] = False
    return train_mask


def _compute_class_means(X_train, y_train, num_classes):
    """
    Mean prototype per class: [C, F]
    """
    device = X_train.device
    dtype = X_train.dtype
    F = X_train.size(1)

    sums = torch.zeros(num_classes, F, device=device, dtype=dtype)
    counts = torch.zeros(num_classes, 1, device=device, dtype=dtype)

    sums.index_add_(0, y_train, X_train)
    counts.index_add_(0, y_train, torch.ones(y_train.size(0), 1, device=device, dtype=dtype))

    counts = counts.clamp_min(1.0)
    means = sums / counts
    return means


def _compute_class_binary_prototypes(X_train, y_train, num_classes, threshold_ratio=0.1):
    """
    For binary/sparse features:
    prototype[c, f] = 1 if fraction of ones in class c at feature f >= threshold_ratio
    """
    device = X_train.device
    dtype = X_train.dtype
    F = X_train.size(1)

    X_bin = (X_train > 0).to(dtype)

    sums = torch.zeros(num_classes, F, device=device, dtype=dtype)
    counts = torch.zeros(num_classes, 1, device=device, dtype=dtype)

    sums.index_add_(0, y_train, X_bin)
    counts.index_add_(0, y_train, torch.ones(y_train.size(0), 1, device=device, dtype=dtype))

    frac = sums / counts.clamp_min(1.0)
    return (frac >= threshold_ratio).to(dtype)


def _compute_class_max_prototypes(X_train, y_train, num_classes):
    """
    Max prototype per class.
    Uses a class loop, but all heavy tensor ops stay on GPU.
    """
    device = X_train.device
    dtype = X_train.dtype
    F = X_train.size(1)

    out = torch.zeros(num_classes, F, device=device, dtype=dtype)
    for c in range(num_classes):
        mask = (y_train == c)
        if mask.any():
            out[c] = X_train[mask].max(dim=0).values
    return out


def Proto_embeddings_cuda_binary(
    data,
    test_idx,
    threshold_ratio=0.1,
    use_max_proto=True,
    use_binary_proto=True,
):
    """
    GPU-friendly proto embeddings for binary/sparse features.

    Returns
    -------
    proto_max_sim   : [N, C] similarity to per-class max prototype (optional)
    proto_bin_sim   : [N, C] similarity to per-class binary-frequency prototype (optional)
    """
    print("Extracting Proto Features...")

    device = data.x.device
    X = data.x
    y = data.y.view(-1).long()

    N, F = X.shape
    C = int(y.max().item()) + 1

    train_mask = _build_train_mask(N, test_idx, device)
    X_train = X[train_mask]
    y_train = y[train_mask]

    outputs = []

    if use_max_proto:
        landmark_max = _compute_class_max_prototypes(X_train, y_train, C)
        proto_max_sim = similarity_torch(X, landmark_max)
        outputs.append(proto_max_sim)

    if use_binary_proto:
        landmark_bin = _compute_class_binary_prototypes(
            X_train, y_train, C, threshold_ratio=threshold_ratio
        )
        proto_bin_sim = similarity_torch(X, landmark_bin)
        outputs.append(proto_bin_sim)

    return tuple(outputs)


def proto_embeddings_euclidean_torch(data, test_idx):
    """
    GPU-friendly Euclidean proto embeddings.

    Returns
    -------
    dist : [N, C]
        Euclidean distance from each node to each class-mean prototype.
    """
    print("Extracting Proto Features...")

    device = data.x.device
    X = data.x
    y = data.y.view(-1).long()

    N, F = X.shape
    C = int(y.max().item()) + 1

    train_mask = _build_train_mask(N, test_idx, device)
    X_train = X[train_mask]
    y_train = y[train_mask]

    class_means = _compute_class_means(X_train, y_train, C)

    # [N, C]
    dist = torch.cdist(X, class_means, p=2)
    return dist

import math
from typing import Optional, Tuple

import torch


# =========================================================
# Basic helpers
# =========================================================
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


def build_sparse_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    undirected: bool = True,
    remove_self_loops: bool = True,
) -> torch.Tensor:
    """
    Build a coalesced sparse adjacency matrix A of shape [N, N].
    """
    row, col = edge_index[0], edge_index[1]

    if remove_self_loops:
        mask = row != col
        row, col = row[mask], col[mask]

    if undirected:
        row2 = torch.cat([row, col], dim=0)
        col2 = torch.cat([col, row], dim=0)
        row, col = row2, col2

    indices = torch.stack([row, col], dim=0).to(device)
    values = torch.ones(indices.size(1), device=device, dtype=torch.float32)

    A = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device)
    A = A.coalesce()

    # turn duplicate edges into 1
    idx = A.indices()
    val = torch.ones(idx.size(1), device=device, dtype=torch.float32)
    A = torch.sparse_coo_tensor(idx, val, (num_nodes, num_nodes), device=device).coalesce()

    return A


# =========================================================
# Exact batched annulus descriptor on GPU
# =========================================================
@torch.no_grad()
def compute_annulus_descriptor_all_nodes_gpu(
    data,
    y_masked: torch.Tensor,
    num_classes: Optional[int] = None,
    max_k: int = 3,
    unknown_label: int = -1,
    undirected: bool = True,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Exact annulus descriptor for ALL nodes using batched GPU BFS.

    For each node v and hop k=1,...,max_k:
        A_k(v)      = nodes at exact shortest-path distance k
        A_k^lab(v)  = labeled nodes in A_k(v)

        p_k(v)[c] = class-c proportion among labeled nodes in A_k(v)
                    or uniform if no labeled nodes exist in A_k(v)

        cov_k(v)  = m_k(v) / (n_k(v)+1)
        l_k(v)    = log(m_k(v)+1)

    Output shape:
        [N, max_k * (num_classes + 2)]
    """
    device = data.x.device
    num_nodes = data.num_nodes

    y_masked = y_masked.to(device).view(-1)
    known_mask = (y_masked != unknown_label)

    if num_classes is None:
        known_y = y_masked[known_mask]
        if known_y.numel() == 0:
            raise ValueError("No known labels available to infer num_classes.")
        num_classes = int(known_y.max().item()) + 1

    # Sparse adjacency
    A = build_sparse_adjacency(
        edge_index=data.edge_index.to(device),
        num_nodes=num_nodes,
        device=device,
        undirected=undirected,
        remove_self_loops=True,
    )

    # One-hot class matrix for known labels only: [N, C]
    class_onehot = torch.zeros(num_nodes, num_classes, device=device, dtype=dtype)
    if known_mask.any():
        known_idx = torch.where(known_mask)[0]
        class_onehot[known_idx, y_masked[known_idx].long()] = 1.0

    known_float = known_mask.to(dtype).unsqueeze(1)  # [N, 1]

    block_dim = num_classes + 2
    out_dim = max_k * block_dim
    desc = torch.empty((num_nodes, out_dim), device=device, dtype=dtype)

    uniform = torch.full((num_classes,), 1.0 / num_classes, device=device, dtype=dtype)

    # Process source nodes in chunks
    for start in range(0, num_nodes, batch_size):
        end = min(start + batch_size, num_nodes)
        B = end - start
        src = torch.arange(start, end, device=device)

        # frontier, visited: [B, N]
        frontier = torch.zeros(B, num_nodes, device=device, dtype=torch.bool)
        frontier[torch.arange(B, device=device), src] = True

        visited = frontier.clone()

        blocks_per_k = []

        for _k in range(1, max_k + 1):
            # neighbors of current frontier
            # sparse.mm: [N,N] x [N,B] -> [N,B], then transpose -> [B,N]
            nbr_counts = torch.sparse.mm(A, frontier.to(dtype).t()).t()
            next_frontier = nbr_counts > 0

            # exact annulus = newly reached nodes only
            next_frontier = next_frontier & (~visited)

            visited |= next_frontier
            frontier = next_frontier

            ring = frontier.to(dtype)  # [B, N]

            # n_k(v): total nodes in exact k-annulus
            nk = ring.sum(dim=1, keepdim=True)  # [B,1]

            # m_k(v): labeled nodes in exact k-annulus
            mk = ring @ known_float  # [B,1]

            # counts by class in exact k-annulus
            counts = ring @ class_onehot  # [B,C]

            # p_k(v)
            pk = counts / mk.clamp_min(1.0)
            no_labeled = (mk.squeeze(1) == 0)
            if no_labeled.any():
                pk[no_labeled] = uniform

            cov_k = mk / (nk + 1.0)
            ell_k = torch.log(mk + 1.0)

            block = torch.cat([pk, cov_k, ell_k], dim=1)  # [B, C+2]
            blocks_per_k.append(block)

        desc[start:end] = torch.cat(blocks_per_k, dim=1)

    return desc


def build_descriptor_from_split_indices_gpu(
    data,
    test_idx,
    val_idx=None,
    hide_test_only: bool = True,
    max_k: int = 3,
    unknown_label: int = -1,
    num_classes: Optional[int] = None,
    undirected: bool = True,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    GPU version of descriptor construction.
    """
    device = data.x.device

    if not torch.is_tensor(test_idx):
        test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)
    else:
        test_idx = test_idx.to(device).long()

    if hide_test_only:
        hide_idx = test_idx
    else:
        if val_idx is None:
            raise ValueError("val_idx must be provided when hide_test_only=False.")
        if not torch.is_tensor(val_idx):
            val_idx = torch.tensor(val_idx, dtype=torch.long, device=device)
        else:
            val_idx = val_idx.to(device).long()
        hide_idx = torch.cat([val_idx, test_idx], dim=0)

    y_masked = hide_labels_by_index(
        y=data.y.view(-1).to(device),
        hide_idx=hide_idx,
        unknown_label=unknown_label,
    )

    desc = compute_annulus_descriptor_all_nodes_gpu(
        data=data,
        y_masked=y_masked,
        num_classes=num_classes,
        max_k=max_k,
        unknown_label=unknown_label,
        undirected=undirected,
        dtype=dtype,
        batch_size=batch_size,
    )

    return desc, y_masked