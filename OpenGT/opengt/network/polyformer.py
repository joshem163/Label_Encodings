from types import SimpleNamespace

import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import GeneralLayer, new_layer_config
from torch_geometric.graphgym.register import register_network

from opengt.encoder.feature_encoder import FeatureEncoder
from opengt.layer.polyformer_layer import PolyFormerBlock


@register_network("PolyFormer")
class PolyFormer(nn.Module):
    """
    PolyFormer model wrapper for node-level tasks.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        self.pre_mp = GeneralLayer(
            "linear",
            new_layer_config(
                dim_in=dim_in,
                dim_out=cfg.gt.dim_hidden,
                num_layers=1,
                has_act=True,
                has_bias=True,
                cfg=cfg,
            ),
        )

        poly_args = SimpleNamespace(
            K=getattr(cfg.gt, "K", 4),
            base=getattr(cfg.gt, "base", "cheby"),
            hidden=cfg.gt.dim_hidden,
            n_head=cfg.gt.n_heads,
            multi=getattr(cfg.gt, "multi", 2.0),
            q=getattr(cfg.gt, "q", 1.0),
            dprate=cfg.gt.dropout,
            d_ffn=getattr(cfg.gt, "d_ffn", cfg.gt.dim_hidden * 2),
        )
        self.token_count = poly_args.K + 1
        self.blocks = nn.ModuleList(
            [PolyFormerBlock(None, poly_args) for _ in range(cfg.gt.layers)]
        )
        self.token_pool = getattr(cfg.gt, "token_pool", "mean")

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gt.dim_hidden, dim_out=dim_out)

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.token_pool == "last":
            return tokens[:, -1, :]
        return tokens.mean(dim=1)

    def forward(self, batch):
        batch = self.encoder(batch)
        batch = self.pre_mp(batch)

        tokens = batch.x.unsqueeze(1).repeat(1, self.token_count, 1)
        for block in self.blocks:
            tokens = block(tokens)

        out_batch = batch.clone()
        out_batch.x = self._pool_tokens(tokens)
        out_batch = self.post_mp(out_batch)
        return out_batch
