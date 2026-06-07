from types import SimpleNamespace

import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, BatchNorm1dNode
from torch_geometric.graphgym.register import register_layer


@register_layer("feature_encoder")
class FeatureEncoder(torch.nn.Module):
    """
    Encodes node and edge features.
    Receives the encoder type from the config file. 

    Parameters:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        self.use_lss = bool(getattr(cfg.gt, 'use_lss', False))
        self.use_proto = bool(getattr(cfg.gt, 'use_proto', False))
        self._expected_aug_dim = 0

        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner)
            if cfg.dataset.node_encoder_bn:
                self.node_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))
            # Update dim_in to reflect the new dimension of the node features
            self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(
                    new_layer_config(cfg.gnn.dim_edge, -1, -1, has_act=False,
                                     has_bias=False, cfg=cfg))

        # Match Label_Encodings fusion style: x <- [x || proto || lss].
        # The first downstream layer needs the concatenated input width ahead of time.
        if self.use_lss or self.use_proto:
            num_classes = int(getattr(cfg.share, 'dim_out', 0))
            if num_classes <= 0:
                raise ValueError("cfg.share.dim_out must be set when gt.use_proto/use_lss is enabled.")

            # Proto width follows the reference behavior:
            # binary features -> 2C (Fec + SFec), otherwise -> C.
            binary_hint = self._dataset_binary_hint(str(getattr(cfg.dataset, 'name', '')))
            proto_dim = 0
            if self.use_proto:
                proto_dim = (2 * num_classes) if binary_hint else num_classes

            lss_dim = 0
            if self.use_lss:
                max_k = int(getattr(cfg.gt, 'lss_max_k', 3))
                lss_dim = max_k * (num_classes + 2)

            self._expected_aug_dim = proto_dim + lss_dim
            self.dim_in = self.dim_in + self._expected_aug_dim

    @staticmethod
    def _mask_to_idx(mask, device):
        if mask is None:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.where(mask)[0].to(device=device, dtype=torch.long)

    @staticmethod
    def _dataset_binary_hint(dataset_name: str) -> bool:
        # Heuristic for common node-classification benchmarks used in this repo.
        non_binary = {
            'pubmed',
            'coauthor-cs',
            'coauthor-physics',
            'wikics',
            'corafull',
            'flickr',
            'blogcatalog',
        }
        return dataset_name.lower() not in non_binary

    @staticmethod
    def _dense_2d(x: torch.Tensor) -> torch.Tensor:
        # Downstream fusion assumes dense, row-major 2D node features.
        if x.layout != torch.strided:
            x = x.to_dense()
        return x

    def _ensure_split_cache(self, batch):
        # Cache split indices to avoid recreating tensors every forward pass.
        if not hasattr(batch, '_split_idx_cache'):
            device = batch.x.device
            train_idx = self._mask_to_idx(getattr(batch, 'train_mask', None), device)
            val_idx = self._mask_to_idx(getattr(batch, 'val_mask', None), device)
            test_idx = self._mask_to_idx(getattr(batch, 'test_mask', None), device)
            batch._split_idx_cache = (train_idx, val_idx, test_idx)
        return batch._split_idx_cache

    def _maybe_add_lss_proto(self, batch, raw_x):
        if not (self.use_lss or self.use_proto):
            return batch

        train_idx, val_idx, test_idx = self._ensure_split_cache(batch)
        if test_idx.numel() == 0:
            return batch

        if self.use_lss and not hasattr(batch, '_lss_desc_cache'):
            from lss_descriptors import build_descriptor_from_split_indices

            desc, _ = build_descriptor_from_split_indices(
                data=batch,
                test_idx=test_idx,
                val_idx=val_idx,
                hide_test_only=bool(getattr(cfg.gt, 'lss_hide_test_only', False)),
                max_k=int(getattr(cfg.gt, 'lss_max_k', 3)),
                backend=str(getattr(cfg.gt, 'lss_backend', 'auto')),
                gpu_batch_size=int(getattr(cfg.gt, 'lss_gpu_batch_size', 256)),
            )
            batch._lss_desc_cache = desc.to(batch.x.device, dtype=batch.x.dtype)

        if self.use_proto and not hasattr(batch, '_proto_desc_cache'):
            from proto_features import build_proto_descriptor

            proto_data = SimpleNamespace(
                x=raw_x,
                y=batch.y,
                edge_index=batch.edge_index,
            )
            proto = build_proto_descriptor(
                data=proto_data,
                dataset_name=cfg.dataset.name,
                test_idx=test_idx,
                val_idx=val_idx,
                hide_test_only=bool(getattr(cfg.gt, 'proto_hide_test_only', False)),
                ir_squirrel=float(getattr(cfg.gt, 'proto_ir_squirrel', 0.01)),
                ir_other=float(getattr(cfg.gt, 'proto_ir_other', 0.1)),
            )
            batch._proto_desc_cache = proto.to(batch.x.device, dtype=batch.x.dtype)

        parts = []
        if self.use_proto:
            parts.append(batch._proto_desc_cache)
        if self.use_lss:
            parts.append(batch._lss_desc_cache)
        if parts:
            parts = [self._dense_2d(p) for p in parts]
            aug = torch.cat(parts, dim=1)
            if self._expected_aug_dim and aug.size(1) != self._expected_aug_dim:
                if aug.size(1) < self._expected_aug_dim:
                    pad = torch.zeros(
                        aug.size(0),
                        self._expected_aug_dim - aug.size(1),
                        device=aug.device,
                        dtype=aug.dtype,
                    )
                    aug = torch.cat([aug, pad], dim=1)
                else:
                    aug = aug[:, :self._expected_aug_dim]
            base_x = self._dense_2d(batch.x)
            batch.x = torch.cat([base_x, aug], dim=1)
        return batch

    def forward(self, batch):
        raw_x = batch.x
        for module in self.children():
            batch = module(batch)
        batch = self._maybe_add_lss_proto(batch, raw_x)
        return batch
