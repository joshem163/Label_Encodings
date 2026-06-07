import csv
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor


def _to_index_tensor(index, device):
    if not torch.is_tensor(index):
        index = torch.tensor(index, dtype=torch.long)
    return index.to(device).long()


def _build_train_mask(n, device, test_idx, val_idx=None, hide_test_only=False):
    """Create train mask by excluding test nodes (and val nodes unless hidden)."""
    test_idx = _to_index_tensor(test_idx, device)
    train_mask = torch.ones(n, dtype=torch.bool, device=device)
    train_mask[test_idx] = False
    if not hide_test_only and val_idx is not None:
        val_idx = _to_index_tensor(val_idx, device)
        train_mask[val_idx] = False
    return train_mask


def is_binary_feature_matrix(x: torch.Tensor) -> bool:
    """Return True when all feature values are exactly 0/1."""
    if x.numel() == 0:
        return False
    return bool(torch.all((x == 0) | (x == 1)).item())


def spatial_embeddings_cuda(data, test_index):
    device = data.y.device
    y = data.y.view(-1).to(device)
    n = y.size(0)

    c = int(y.max().item()) + 2
    test_class = c - 1

    test_index = _to_index_tensor(test_index, device)

    label = y.clone()
    label[test_index] = test_class

    onehot = F.one_hot(label, num_classes=c).to(torch.float32)

    edge_index = data.edge_index.to(device)
    adj1 = SparseTensor.from_edge_index(edge_index, sparse_sizes=(n, n)).to(device)
    hop1 = adj1.matmul(onehot)

    mask_no_self = edge_index[0] != edge_index[1]
    edge_index2 = edge_index[:, mask_no_self]
    adj2 = SparseTensor.from_edge_index(edge_index2, sparse_sizes=(n, n)).to(device)
    hop2 = adj1.matmul(adj2.matmul(onehot))

    return torch.cat([hop1, hop2], dim=1)


def cosine_similarity_torch(a, b, eps=1e-12):
    dot = a @ b.t()
    an = a.norm(p=2, dim=1, keepdim=True)
    bn = b.norm(p=2, dim=1, keepdim=True)
    denom = an * bn.t()
    return torch.where(denom > 0, dot / (denom + eps), torch.zeros_like(dot))


def similarity_torch(a, b, threshold=0.0):
    a_bin = (a > threshold).to(torch.float32)
    b_bin = (b > threshold).to(torch.float32)
    sim = a_bin @ b_bin.t()
    return sim


def Proto_embeddings_cuda(
    data,
    dataset_name,
    test_idx,
    val_idx=None,
    hide_test_only=False,
    ir_squirrel=0.01,
    ir_other=0.1,
):
    """
    Compute binary-compatible proto features using train-only landmarks.
    Returns:
    - f_max: [N, C] similarity to per-class max prototype
    - f_bin: [N, C] similarity to per-class thresholded-binary prototype
    """
    device = data.x.device
    x = data.x
    y = data.y.view(-1).long()
    n, feat_dim = x.shape
    num_classes = int(y.max().item()) + 1

    ir = ir_squirrel if dataset_name == "squirrel" else ir_other
    train_mask = _build_train_mask(
        n=n,
        device=device,
        test_idx=test_idx,
        val_idx=val_idx,
        hide_test_only=hide_test_only,
    )

    x_train = x[train_mask]
    y_train = y[train_mask]

    landmark_max = torch.zeros(num_classes, feat_dim, device=device, dtype=x.dtype)
    landmark_bin = torch.zeros(num_classes, feat_dim, device=device, dtype=x.dtype)

    for cls in range(num_classes):
        cls_mask = (y_train == cls)
        if cls_mask.any():
            xc = x_train[cls_mask]
            landmark_max[cls] = xc.max(dim=0).values
            frac_ones = (xc == 1).to(x.dtype).mean(dim=0)
            landmark_bin[cls] = (frac_ones >= ir).to(x.dtype)

    f_max = similarity_torch(x, landmark_max)
    f_bin = similarity_torch(x, landmark_bin)
    return f_max, f_bin


def Proto_embeddings_cuda_binary(data, dataset_name, test_idx, Ir_squirrel=0.01, Ir_other=0.1):
    """
    CUDA-compatible Proto_embeddings using cosine similarity.
    Returns:
    Fec : [N, C] cosine similarity to landmark1 (per-class max prototype)
    SFec : [N, C] cosine similarity to landmark2 (per-class thresholded-ones prototype)
    """
    print("Extracting Proto Features.....")
    device = data.x.device
    X = data.x
    y = data.y.view(-1).long()
    N, F = X.shape
    C = int(y.max().item()) + 1
    Ir = Ir_squirrel if dataset_name == "squirrel" else Ir_other

    if not torch.is_tensor(test_idx):
        test_idx = torch.tensor(test_idx, dtype=torch.long)
    test_idx = test_idx.to(device)

    train_mask = torch.ones(N, dtype=torch.bool, device=device)
    train_mask[test_idx] = False
    X_train = X[train_mask]
    y_train = y[train_mask]

    landmark1 = torch.zeros(C, F, device=device, dtype=X.dtype)
    landmark2 = torch.zeros(C, F, device=device, dtype=X.dtype)

    for cls in range(C):
        cls_mask = (y_train == cls)
        if cls_mask.any():
            Xc = X_train[cls_mask]
            landmark1[cls] = Xc.max(dim=0).values
            frac_ones = (Xc == 1).to(X.dtype).mean(dim=0)
            landmark2[cls] = (frac_ones >= Ir).to(X.dtype)
        else:
            landmark1[cls].zero_()
            landmark2[cls].zero_()

    Fec = similarity_torch(X, landmark1)
    SFec = similarity_torch(X, landmark2)
    return Fec, SFec


def proto_embeddings_euclidean_torch(data, test_idx):
    """
    Compute Euclidean proto features using class-mean train landmarks.
    Returns:
    - dist: [N, C] Euclidean distance to class landmarks
    """
    device = data.x.device
    x = data.x
    y = data.y.view(-1).long()
    N, F = x.size(0), x.size(1)
    C = int(y.max().item()) + 1

    if not torch.is_tensor(test_idx):
        test_idx = torch.tensor(test_idx, dtype=torch.long)
    test_idx = test_idx.to(device)

    train_mask = torch.ones(N, dtype=torch.bool, device=device)
    train_mask[test_idx] = False

    landmarks = torch.zeros(C, F, device=device, dtype=x.dtype)
    counts = torch.zeros(C, device=device, dtype=x.dtype)
    x_train = x[train_mask]
    y_train = y[train_mask]

    landmarks.index_add_(0, y_train, x_train)
    counts.index_add_(0, y_train, torch.ones_like(y_train, dtype=x.dtype))
    counts = counts.clamp_min(1.0).unsqueeze(1)
    landmarks = landmarks / counts

    print("Extracting Proto Features")
    dist = torch.cdist(x, landmarks, p=2)
    return dist


def build_proto_descriptor(
    data,
    dataset_name,
    test_idx,
    val_idx=None,
    hide_test_only=False,
    ir_squirrel=0.01,
    ir_other=0.1,
):
    """
    Build proto descriptor with dataset-aware binary/non-binary handling.
    Binary x -> [f_max, f_bin]
    Non-binary x -> [euclidean_dist]
    """
    if not torch.is_tensor(test_idx):
        test_idx_eff = torch.tensor(test_idx, dtype=torch.long)
    else:
        test_idx_eff = test_idx.long()

    if not hide_test_only:
        if val_idx is None:
            raise ValueError("val_idx must be provided when hide_test_only=False.")
        if not torch.is_tensor(val_idx):
            val_idx_t = torch.tensor(val_idx, dtype=torch.long)
        else:
            val_idx_t = val_idx.long()
        test_idx_eff = torch.unique(torch.cat([val_idx_t, test_idx_eff], dim=0))

    # Proto feature routines expect dense tensor ops (comparisons, cdist).
    # Some datasets (e.g., Flickr) provide sparse CSR node features.
    x = data.x
    if x.is_sparse or x.layout in (torch.sparse_coo, torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
        dense_data = SimpleNamespace(
            x=x.to_dense(),
            y=data.y,
            edge_index=data.edge_index,
        )
    else:
        dense_data = data

    if is_binary_feature_matrix(dense_data.x):
        f1, f2 = Proto_embeddings_cuda_binary(
            data=dense_data,
            dataset_name=dataset_name,
            test_idx=test_idx_eff,
            Ir_squirrel=ir_squirrel,
            Ir_other=ir_other,
        )
        return torch.cat([f1, f2], dim=1)

    f1 = proto_embeddings_euclidean_torch(
        data=dense_data,
        test_idx=test_idx_eff,
    )
    return f1


def save_best_result(dataset_name, best_result, std, best_args, runtime):
    if torch.is_tensor(best_result):
        best_result = best_result.item()
    if torch.is_tensor(std):
        std = std.item()
    if torch.is_tensor(runtime):
        runtime = runtime.item()

    results_file = "best_results.csv"
    file_exists = os.path.isfile(results_file)

    with open(results_file, mode="a", newline="") as f:
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
                "runtime_sec",
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
            round(runtime, 2),
        ])
