import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .gnn.graph import AdjacencyMatrix


class Model(nn.Module):
    
    def __init__(self, seq_len, pred_len, enc_in, node_dim, hidden_dim,  
                 node_proj=True, nonlinear_pred=True, norm=False, mix=True, 
                 adj="unidirected", **kwargs):
        super(Model, self).__init__()
        
        if norm:
            self.norm = nn.InstanceNorm1d(enc_in)
        else:
            self.norm = None

        if mix:
            self.adj = AdjacencyMatrix(adj, **kwargs)
        else:
            self.adj = None

        if node_proj:
            self.node_proj = nn.Sequential(
                nn.Linear(enc_in, node_dim),
                nn.Tanh(),
                nn.Linear(node_dim, node_dim),
                nn.Tanh(),
                nn.Linear(node_dim, enc_in),
            )
        else:
            self.node_proj = None

        if nonlinear_pred:
            self.pred_proj = nn.Sequential(
                nn.Linear(seq_len, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, pred_len),   
            )
        else:
            self.pred_proj = nn.Linear(seq_len, pred_len)

            

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.norm is not None:
            x = self.norm(x.permute(0,2,1)).permute(0,2,1)
        
        if self.adj is not None:
            x = torch.einsum("bld,dn->bln", x, self.adj(x))

        if self.node_proj is not None:
            x = self.node_proj(x)

        pred = self.pred_proj(x.permute(0,2,1)).permute(0,2,1)

            
        return pred
