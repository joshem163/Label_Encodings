
import torch
import torch.nn.functional as F
import networkx as nx
from networkx import ego_graph
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor,matmul
import scipy.sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops

import torch.optim as optim
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv,GATConv, JumpingKnowledge,TransformerConv


#from logger import Logger
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        # return x.log_softmax(dim=-1)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, heads):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.dropout = dropout

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * heads))

        # Final layer (concat=False for classification output)
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1, concat=False))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

class MLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters_mlp(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            # x = F.relu(x)
            x = F.sigmoid(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        # return torch.log_softmax(x, dim=-1)
        return x

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class MLP_H2GCN(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP_H2GCN, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
#             x = data.graph['node_feat']
            x = data.x
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class H2GCN(nn.Module):
    """ our implementation """

    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                 num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                 use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP_H2GCN(in_channels, hidden_channels,
                                 hidden_channels, num_layers=num_mlp_layers, dropout=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers - 2:
                self.bns.append(nn.BatchNorm1d(hidden_channels * 2 * len(self.convs)))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout  # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels * (2 ** (num_layers + 1) - 1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes

        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)

        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)

    def forward(self, data):
        #         x = data.graph['node_feat']
        #         n = data.graph['num_nodes']
        x = data.x
        n = len(data.y)

        adj_t = self.adj_t
        adj_t2 = self.adj_t2

        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2)
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

class ProtoHomophilyGate(nn.Module):
    """
    Computes a per-node gate h_v in (0, 1) from proto-embedding cosine
    similarity aggregated over each node's incoming neighborhood.

    For each node v:
        s(u, v) = cosine_similarity(alpha(u), alpha(v))  for u in N(v)
        h_v     = sigmoid( temperature * mean_{u in N(v)} s(u, v) )

    h_v ~ 1  =>  neighbors are semantically similar to v
             =>  homophilic neighborhood  =>  trust GNN branch
    h_v ~ 0  =>  neighbors are semantically dissimilar to v
             =>  heterophilic neighborhood  =>  trust proto branch

    Convention note (PyG):
        edge_index[0] = source nodes (u)
        edge_index[1] = target nodes (v)
    Aggregation is over incoming edges, i.e. we scatter over edge_index[1].

    Args:
        learnable_temperature: if True, the sigmoid temperature is a learned
            scalar parameter (recommended). If False, fixed at 1.0.
    """

    def __init__(self, learnable_temperature: bool = True):
        super().__init__()
        if learnable_temperature:
            # Initialised to 1.0; the model can sharpen or flatten the gate.
            # Monitor this during training: if it collapses to ~0 the gate
            # is becoming uniform, which may indicate a problem.
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('temperature', torch.ones(1))

    def forward(
        self,
        alpha: torch.Tensor,       # [N, proto_dim]
        edge_index: torch.Tensor,  # [2, E]  (source, target) convention
    ) -> torch.Tensor:             # [N]  gate values in (0, 1)

        N = alpha.size(0)

        # Add self-loops so isolated nodes still get a gate value
        # (their own proto-embedding similarity with themselves = 1.0,
        #  so isolated nodes default to h_v = sigmoid(temperature) ~ 0.73)
        ei, _ = add_self_loops(edge_index, num_nodes=N)
        src, tgt = ei[0], ei[1]   # src = u (source), tgt = v (target)

        # L2-normalise proto-embeddings for cosine similarity
        alpha_norm = F.normalize(alpha, p=2, dim=-1)   # [N, proto_dim]

        # Cosine similarity along each edge: s(src_i, tgt_i)
        sim = (alpha_norm[src] * alpha_norm[tgt]).sum(dim=-1)  # [E]

        # Mean incoming similarity per target node v
        # scatter_add over tgt (edge_index[1]) = incoming aggregation
        sim_sum = torch.zeros(N, device=alpha.device)
        sim_sum.scatter_add_(0, tgt, sim)

        count = torch.zeros(N, device=alpha.device)
        count.scatter_add_(0, tgt, torch.ones_like(sim))

        mean_sim = sim_sum / count.clamp(min=1.0)   # [N]

        # Gate with learnable temperature
        h = torch.sigmoid(self.temperature * mean_sim)  # [N]
        return h
class ProtoGated(nn.Module):
    """
    Proto-Gated GraphSAGE: homophily-aware fusion of a GraphSAGE branch
    (raw features + graph structure) and a Proto-MLP branch
    (proto-embeddings, no graph structure).

    Forward pass:
        H_gnn   = SAGE(X, G)                              GNN branch
        H_proto = MLP(alpha)                              Proto branch
        h_v     = sigmoid(T * mean_neighbor_cosine(alpha))  Gate
        z_v     = h_v * H_gnn(v) + (1 - h_v) * H_proto(v)
        y_hat_v = softmax(Linear(z_v))

    Both branches output vectors of size hidden_dim with no final activation,
    so the fusion is a convex combination of comparable representations.
    A shared ReLU is applied after fusion before the classifier.

    Args:
        raw_dim              : dimensionality of raw node features X
        proto_dim            : dimensionality of proto-embeddings alpha
        hidden_dim           : hidden size shared by both branches
        output_dim           : number of output classes
        num_layers           : depth of both branches (kept equal for fairness)
        dropout              : dropout rate applied in both branches and
                               after fusion
        learnable_temperature: whether gate temperature is a learned parameter
    """

    def __init__(
        self,
        raw_dim: int,
        proto_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        gnn_type: str,
        heads: int,
        learnable_temperature: bool = True,

    ):
        super().__init__()

        # Both branches map to hidden_dim (no activation on final layer)
        if gnn_type == 'GCN':
            self.gnn_branch = GCN(
                raw_dim, hidden_dim, hidden_dim, num_layers, dropout
            )
        elif gnn_type=='SAGE':
            self.gnn_branch = SAGE(
                raw_dim, hidden_dim, hidden_dim, num_layers, dropout
            )
        elif gnn_type=='GAT':
            self.gnn_branch   = GAT(
            raw_dim, hidden_dim, hidden_dim, num_layers, dropout,heads)
        else:
            print("Change the GNN type..")
        self.proto_branch = MLP(
            proto_dim, hidden_dim, hidden_dim, num_layers, dropout
        )

        # Gate derived purely from proto-embedding geometry
        self.gate = ProtoHomophilyGate(learnable_temperature)

        # Shared classifier on top of fused representation
        self.classifier = nn.Linear(2*hidden_dim, output_dim)
        self.dropout    = dropout

    def forward(
        self,
        x: torch.Tensor,           # [N, raw_dim]    raw node features
        alpha: torch.Tensor,       # [N, proto_dim]  proto-embeddings
        edge_index: torch.Tensor,  # [2, E]
    ) -> torch.Tensor:             # [N, output_dim] log-probabilities

        # --- GNN branch: message passing on raw features ---
        # Output: [N, hidden_dim], no final activation inside branch
        H_gnn = self.gnn_branch(x, edge_index)

        # --- Proto branch: MLP on proto-embeddings, no graph ---
        # Output: [N, hidden_dim], no final activation inside branch
        H_proto = self.proto_branch(alpha)

        # --- Gate: derived purely from proto-embedding geometry ---
        # h_v in (0,1): high = homophilic = trust GNN
        #               low  = heterophilic = trust proto
        #h = self.gate(alpha, edge_index).unsqueeze(-1)  # [N, 1] for broadcast
        # print(h)

        # --- Gated fusion (convex combination) ---
        #z = h * H_gnn + (1.0 - h) * H_proto             # [N, hidden_dim]
        #z = h * H_proto + (1.0 - h) * H_gnn
        z=torch.cat([H_gnn, H_proto], dim=1)

        # Single ReLU + dropout after fusion, before classifier
        z = F.relu(z)
        z = F.dropout(z, p=self.dropout, training=self.training)

        # --- Classification ---
        out = self.classifier(z)                         # [N, output_dim]
        return F.log_softmax(out, dim=-1)

    @torch.no_grad()
    def get_gate_values(
        self,
        alpha: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:             # [N] gate values in (0, 1)
        """
        Returns per-node gate values h_v for analysis and visualization.
        Call after training to inspect which nodes were detected as
        heterophilic vs homophilic by the proto-geometry signal.

        Suggested use:
            h = model.get_gate_values(alpha, data.edge_index).cpu().numpy()
            # Correlate h with per-node ground-truth homophily for analysis
        """
        self.eval()
        return self.gate(alpha, edge_index)


