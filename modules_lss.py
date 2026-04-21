import math
from collections import deque
from typing import Optional, Tuple

import torch
from sklearn.cluster import KMeans
def cosine_similarity_torch(A, B, eps=1e-12):
    """
    Vectorized cosine similarity matching your numpy version:
      cos(a,b) = dot(a,b) / (||a|| * ||b||)
      if ||a||==0 or ||b||==0 -> 0

    A: [N, F]
    B: [C, F]
    returns: [N, C]
    """
    # dot products: [N, C]
    dot = A @ B.t()

    # norms: [N, 1], [C, 1]
    An = A.norm(p=2, dim=1, keepdim=True)
    Bn = B.norm(p=2, dim=1, keepdim=True)

    denom = An * Bn.t()  # [N, C]

    # If denom == 0 -> similarity = 0 (exactly like your function)
    sim = torch.where(denom > 0, dot / denom, torch.zeros_like(dot))
    return sim

def similarity_torch(A, B, threshold=0.0):
    """
    Vectorized intersection similarity (counts of logical AND):
      sim(i,j) = sum_k [A[i,k] & B[j,k]]

    A: [N, F]
    B: [C, F]
    returns: [N, C]   (integer counts as float tensor)

    Notes:
    - If A/B are already boolean or {0,1}, set threshold=0.0 and it works.
    - If A/B are real-valued, this binarizes with (x > threshold).
    """
    A_bin = (A > threshold)
    B_bin = (B > threshold)

    # Convert to int for matrix multiply: AND count = sum(A_bin * B_bin)
    A_int = A_bin.to(torch.int32)
    B_int = B_bin.to(torch.int32)

    # [N, C] = [N, F] @ [F, C]
    sim = A_int @ B_int.t()

    # return float (often nicer downstream), but you can keep int if you want
    return sim.to(torch.float32)
def build_adjacency_list(edge_index: torch.Tensor, num_nodes: int, undirected: bool = True):
    adj = [[] for _ in range(num_nodes)]
    row, col = edge_index.cpu()

    for u, v in zip(row.tolist(), col.tolist()):
        adj[u].append(v)
        if undirected and u != v:
            adj[v].append(u)

    return adj


def exact_k_hop_annuli(adj, source: int, max_k: int = 3):
    """
    annuli[k] = nodes at exact shortest-path distance k from source
    """
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
    """
    Replace labels at hide_idx by unknown_label.
    """
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
    """
    Compute descriptor for ALL nodes using only known labels in y_masked.

    For each node v and hop k=1,...,max_k:
        A_k(v)      = {u : dist(u,v)=k}
        A_k^lab(v)  = {u in A_k(v) : y_masked[u] != unknown_label}
        n_k(v)      = |A_k(v)|
        m_k(v)      = |A_k^lab(v)|

        p_k(v)[c] = count of class c in A_k^lab(v) / max(m_k(v),1)
                    uniform if m_k(v)=0

        cov_k(v) = m_k(v) / (n_k(v)+1)
        l_k(v)   = log(m_k(v)+1)

    Output:
        [num_nodes, max_k * (num_classes + 2)]
    """
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build descriptor for all nodes using explicit split indices.

    Args:
        data: PyG Data object
        test_idx: indices of test nodes whose labels must be hidden
        val_idx: validation indices, only used if hide_test_only=False
        hide_test_only:
            True  -> hide only test labels
            False -> hide both validation and test labels
        max_k: maximum hop radius
        unknown_label: dummy label for hidden nodes
        num_classes: number of classes
        undirected: treat graph as undirected
        dtype: output dtype

    Returns:
        desc: [N, max_k * (num_classes + 2)]
        y_masked: [N]
    """
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

def Proto_embeddings_cuda_binary(data, dataset_name, test_idx, Ir_squirrel=0.01, Ir_other=0.1):
    """
    CUDA-compatible Proto_embeddings using cosine similarity.

    Returns:
        Fec  : [N, C] cosine similarity to landmark1 (per-class max prototype)
        SFec : [N, C] cosine similarity to landmark2 (per-class thresholded-ones prototype)
    """
    print("Extracting Proto Features.....")
    device = data.x.device
    X = data.x                      # [N, F] (GPU/CPU)
    y = data.y.view(-1).long()      # [N]
    N, F = X.shape
    C = int(y.max().item()) + 1

    Ir = Ir_squirrel if dataset_name == "squirrel" else Ir_other

    # test_idx -> torch tensor on device
    if not torch.is_tensor(test_idx):
        test_idx = torch.tensor(test_idx, dtype=torch.long)
    test_idx = test_idx.to(device)

    # train mask: exclude test nodes
    train_mask = torch.ones(N, dtype=torch.bool, device=device)
    train_mask[test_idx] = False

    X_train = X[train_mask]
    y_train = y[train_mask]

    # landmarks
    landmark1 = torch.zeros(C, F, device=device, dtype=X.dtype)  # per-class max
    landmark2 = torch.zeros(C, F, device=device, dtype=X.dtype)  # per-class thresholded ones

    #num_cluster=len(np.unique(data.y))
    # num_cluster=100
    # kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X_train)
    # landmark3 = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    for cls in range(C):
        cls_mask = (y_train == cls)
        if cls_mask.any():
            Xc = X_train[cls_mask]                         # [Nc, F]
            landmark1[cls] = Xc.max(dim=0).values          # max per feature

            frac_ones = (Xc == 1).to(X.dtype).mean(dim=0)  # fraction of ones per feature
            landmark2[cls] = (frac_ones >= Ir).to(X.dtype) # 0/1 vector
        else:
            # class absent in train split -> zero prototypes (safe)
            landmark1[cls].zero_()
            landmark2[cls].zero_()

    # cosine similarities (vectorized)
    # Fec  = cosine_similarity_torch(X, landmark1)  # [N, C]
    # SFec = cosine_similarity_torch(X, landmark2)  # [N, C]
    Fec  = similarity_torch(X, landmark1)  # [N, C]
    SFec = similarity_torch(X, landmark2)  # [N, C]
    # knnFec = cosine_similarity_torch(X, landmark3)  # [N, C]

    return Fec, SFec


def proto_embeddings_euclidean_torch(data, test_idx):
    """
    CUDA-friendly replacement for proto_embeddings_eucledian.

    Returns:
      dist:  [N, C] torch.FloatTensor on same device as data.x
      cont_fe: whatever Contextual returns (see note)
    """
    device = data.x.device
    x = data.x                      # [N, F] on GPU/CPU
    y = data.y.view(-1).long()      # [N]

    N, F = x.size(0), x.size(1)
    C = int(y.max().item()) + 1

    # test_idx -> torch tensor on device
    if not torch.is_tensor(test_idx):
        test_idx = torch.tensor(test_idx, dtype=torch.long)
    test_idx = test_idx.to(device)

    # train mask excludes test nodes
    train_mask = torch.ones(N, dtype=torch.bool, device=device)
    train_mask[test_idx] = False

    # Compute class landmarks (means) using ONLY train nodes
    landmarks = torch.zeros(C, F, device=device, dtype=x.dtype)
    counts = torch.zeros(C, device=device, dtype=x.dtype)

    x_train = x[train_mask]
    y_train = y[train_mask]

    # sum features per class
    landmarks.index_add_(0, y_train, x_train)
    counts.index_add_(0, y_train, torch.ones_like(y_train, dtype=x.dtype))

    # avoid divide-by-zero (if a class is absent in train split)
    counts = counts.clamp_min(1.0).unsqueeze(1)     # [C, 1]
    landmarks = landmarks / counts                  # [C, F]

    # num_cluster=100
    print("Extracting Proto Features")
    # kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(x_train)
    # landmark3 = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # Compute Euclidean distances from every node to every landmark:
    # dist[i,j] = ||x_i - landmark_j||
    # Use torch.cdist (GPU-accelerated)
    dist = torch.cdist(x, landmarks, p=2)           # [N, C]
    # knnFec = cosine_similarity_torch(x, landmark3)

    # Contextual features:
    # If your Contextual() uses sklearn/PCA, it's CPU-only.
    # You can keep it CPU-side or rewrite it in torch.
    #cont_fe = Contextual(data,test_idx)  # replace with your version if needed

    return dist


RESULTS_FILE = "best_results.csv"

def save_best_result(dataset_name, best_result, std, best_args, runtime):
    import csv
    import os
    import torch

    RESULTS_FILE = "best_results.csv"

    # convert tensors to Python floats
    if torch.is_tensor(best_result):
        best_result = best_result.item()
    if torch.is_tensor(std):
        std = std.item()
    if torch.is_tensor(runtime):
        runtime = runtime.item()

    file_exists = os.path.isfile(RESULTS_FILE)

    with open(RESULTS_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "dataset",
                "model",
                "best_test_mean",
                "best_test_std",
                "lr",
                "hidden_channels",
                "dropout",
                "num_layers",
                "runs",
                "epochs",
                "runtime_sec"
            ])

        writer.writerow([
            dataset_name,
            best_args["model_type"],
            round(best_result, 4),
            round(std, 4),
            best_args["lr"],
            best_args["hidden_channels"],
            best_args["dropout"],
            best_args["num_layers"],
            best_args["runs"],
            best_args["epochs"],
            round(runtime, 2)
        ])
