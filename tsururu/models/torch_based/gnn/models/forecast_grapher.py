import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import SpatialEmbedding, GFC, Predict
from ..graph import AdjacencyMatrix



class GNNBlock(nn.Module):
    def __init__(self, num_nodes, node_dim, z=32, gcn_depth=2, dropout=0.05, strategy="graph_wavenet", **kwargs):
        super(GNNBlock, self).__init__()

        self.adj = AdjacencyMatrix(strategy=strategy, num_nodes=num_nodes, node_dim=node_dim, **kwargs)
        self.gnn = GFC(z, z, gcn_depth, dropout)
        # self.gnn = mixprop(z, z, gcn_depth, dropout, propalpha)
        # self.gnn = GAT(num_nodes, num_nodes, num_nodes, dropout, 0.2, 1, z)
        # self.gnn = GCN(num_nodes, num_nodes, num_nodes, dropout, z)
        self.gelu = nn.GELU()

        self.norm = nn.LayerNorm(num_nodes)
        self._initialize_weights()
    #
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    #x in(B, conv, N, num_nodes)
    def forward(self, x):
        adj = self.adj(x)

        out = self.gelu(self.gnn(x, adj))
        return self.norm(x + out)


class GNNLayer(nn.Module):
    def __init__(self, num_nodes, node_dim, z=32, k=3, gcn_depth=2, dropout=0.05, strategy="graph_wavenet", **kwargs):
        super(GNNLayer, self).__init__()
        
        self.k = k
        

        self.gnn_blocks = nn.ModuleList()
        for i in range(self.k):
            self.gnn_blocks.append(GNNBlock(num_nodes, node_dim, z, gcn_depth, dropout, strategy, **kwargs))

        self.scaler = nn.Conv2d(1 , z, (1, 1))
        self.group_concatenate = nn.Conv2d(z, num_nodes , (1, num_nodes))

   #(B, N, num_nodes)
    def forward(self, x):
        out = x.unsqueeze(1)
        # (B, z, N, num_nodes)
        out = self.scaler(out)
        for i in range (self.k):
            out = self.gnn_blocks[i](out)

        out = self.group_concatenate(out).squeeze(-1)
        out = out.transpose(2,1)
        out = out + x

        return out

class Model(nn.Module):
    """Forecast Grapher model from the paper [ForecastGrapher: Redefining Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/pdf/2405.18036).

    Args:
        model_params (dict): Dictionary containing parameters for the model:
            - seq_len (int): Temporal length of time series.
            - pred_len (int): Length of prediction.
            - num_nodes (int): Number of nodes/variables to be forecasted.
            - node_dim (int): Hidden dimensionality of a node's temporal features in the adjacency matrix.
            - use_norm (bool): Whether to use norm as in DLinear.
            - e_layers (int): Number of layers.
            - individual (bool): DLinear: a linear layer for each variate(channel) individually.
            - z (int): Scaler.
            - k (int): Number of GNN blocks.
            - gcn_depth (int): Depth of graph convolution.
            - dropout (float): Dropout ratio.
            - strategy (str): Strategy for building the graph. Options:
                "global", "directed", "undirected", "unidirected", "dynamic_attention".
            - kwargs (dict[str, Any]): additional arguments passed to AdjacencyMatrix
    """

    def __init__(self, seq_len, pred_len, num_nodes, node_dim, 
                 use_norm=True, e_layers=2, individual=False, z=32, k=3, 
                 gcn_depth=2, dropout=0.05, strategy="graph_wavenet", **kwargs):
        super(Model, self).__init__()
        
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.use_norm = use_norm


        self.model = nn.ModuleList([GNNLayer(num_nodes, node_dim, z, k, gcn_depth, dropout, strategy, **kwargs) for _ in range(e_layers)])

        self.embedding = SpatialEmbedding(seq_len, num_nodes, num_nodes, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(num_nodes)
        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)
        self.projection = nn.Linear(num_nodes, pred_len, bias=True)
        self.seq2pred = Predict(individual ,num_nodes,
                                seq_len, pred_len, dropout)


    def forecast(self, x):
        # Normalization from Non-stationary Transformer.
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x = self.embedding(x)
        for i in range(self.layer):
            x = self.layer_norm(self.model[i](x))

        #out (B, N, num_nodes)
        out = self.projection(x)
        out = out.transpose(2,1)[:, :, :self.num_nodes]
        # out = self.seq2pred(out.transpose(1, 2)).transpose(1, 2)

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            out = out * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len, 1))
            out = out + \
                      (means[:, 0, :].unsqueeze(1).repeat(
                          1, self.pred_len, 1))

        return out[:, -self.pred_len:, :]

    def forward(self, input):
        history = input.mean(dim=1, keepdim=True)
        output = self.forecast(input - history)
        return output + history


