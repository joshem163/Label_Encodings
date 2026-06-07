import torch
import torch.nn as nn
from torch_geometric.graphgym.register import register_edge_encoder, register_node_encoder


@register_node_encoder("rrwp_linear")
class RRWPLinearNodeEncoder(nn.Module):
    """Project absolute RRWP features and add them to node embeddings."""

    def __init__(self, num_steps, emb_dim):
        super().__init__()
        self.proj = nn.Linear(num_steps, emb_dim)

    def forward(self, batch):
        if not hasattr(batch, "rrwp"):
            raise ValueError("RRWP node features 'rrwp' are required but missing in batch.")
        rrwp = batch.rrwp
        if rrwp.dim() == 1:
            rrwp = rrwp.unsqueeze(-1)
        batch.x = batch.x + self.proj(rrwp)
        return batch


@register_edge_encoder("rrwp_linear")
class RRWPLinearEdgeEncoder(nn.Module):
    """Project relative RRWP features and materialize RRWP relation graph as edges."""

    def __init__(self, num_steps, emb_dim, pad_to_full_graph=False,
                 add_node_attr_as_self_loop=False, fill_value=0.0):
        super().__init__()
        self.proj = nn.Linear(num_steps, emb_dim)
        self.pad_to_full_graph = pad_to_full_graph
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.fill_value = fill_value

    def forward(self, batch):
        if not (hasattr(batch, "rrwp_index") and hasattr(batch, "rrwp_val")):
            raise ValueError("RRWP edge features 'rrwp_index' and 'rrwp_val' are required but missing in batch.")

        rrwp_index = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        if rrwp_val.dim() == 1:
            rrwp_val = rrwp_val.unsqueeze(-1)

        # GRIT consumes edges/edge_attr directly; RRWP defines this relation graph.
        batch.edge_index = rrwp_index
        batch.edge_attr = self.proj(rrwp_val)
        return batch
