import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from ..utils import  MixProp, DilatedInception, LayerNorm
from ..graph import AdjacencyMatrix




class Model(nn.Module):
    """MTGNN model from the paper [Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks](https://arxiv.org/pdf/2005.11650).

    Args:
        model_params (dict): Dictionary containing parameters for the model:
            - seq_len (int): Temporal length of time series.
            - num_nodes (int): Number of nodes/variables to be forecasted.
            - in_dim (int): Number of additional features.
            - pred_len (int): Length of prediction.
            - strategy (str): Strategy for building the graph. Options:
                "global", "directed", "undirected", "unidirected", "dynamic_attention".
            - gcn_true (bool): Whether to apply graph convolutions.
            - gcn_depth (int): Depth of graph convolution.
            - static_feat (None): Static adjacency matrix representing external information, if available.
            - dropout (float): Dropout ratio.
            - subgraph_size (int): Number of connections kept for each node in a subgraph.
            - node_dim (int): Hidden dimensionality of a node's temporal features in the adjacency matrix.
            - dilation_exp (int): Dilation exponential.
            - conv_channels (int): Number of convolution channels.
            - residual_channels (int): Number of residual channels.
            - skip_channels (int): Number of skip channels.
            - end_channels (int): Number of end channels.
            - layers (int): Number of layers.
            - propalpha (float): Propagation alpha.
            - tanhalpha (float): Adjacency matrix alpha.
            - layer_norm_affine (bool): LayerNorm's elementwise_affine value.
            - device (str): Device on which to run the model. Options: "cpu", "cuda".
    """

    def __init__(
        self, 
        seq_len, 
        num_nodes, 
        in_dim, 
        pred_len,
        strategy='unidirected', 
        gcn_true=True, 
        gcn_depth=2, 
        static_feat=None, 
        dropout=0.3, 
        subgraph_size=20, 
        node_dim=40, 
        dilation_exp=1, 
        conv_channels=32, 
        residual_channels=32, 
        skip_channels=64, 
        end_channels=128, 
        layers=3, 
        propalpha=0.05, 
        tanhalpha=3, 
        layer_norm_affine=True
    ):
        
        super(Model, self).__init__()

        self.gcn_true = gcn_true
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.layers = layers
        self.seq_len = seq_len

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        
        self.gc = AdjacencyMatrix(strategy=strategy,
                                n_nodes=num_nodes, 
                                k=subgraph_size, 
                                dim=node_dim, 
                                alpha=tanhalpha, 
                                )

        ker = 7

        self.receptive_field = int(1 + (ker - 1) * (dilation_exp**layers - 1) / (dilation_exp - 1)) if dilation_exp > 1 else layers * (ker - 1) + 1

        filter_conv = DilatedInception(residual_channels, conv_channels, dilation_factor=1)
        self.filter_convs = nn.ModuleList([copy.deepcopy(filter_conv) for _ in range(layers)])

        gate_conv = DilatedInception(residual_channels, conv_channels, dilation_factor=1)
        self.gate_convs = nn.ModuleList([copy.deepcopy(gate_conv) for _ in range(layers)])

        residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))
        self.residual_convs = nn.ModuleList([copy.deepcopy(residual_conv) for _ in range(layers)])
        
        gconv = MixProp(conv_channels, residual_channels, gcn_depth, dropout, propalpha)
        if self.gcn_true:
            self.gconv1 = nn.ModuleList([copy.deepcopy(gconv) for _ in range(layers)])
            self.gconv2 = nn.ModuleList([copy.deepcopy(gconv) for _ in range(layers)])

        del filter_conv, gate_conv, residual_conv, gconv

        self.skip_convs = nn.ModuleList()
        self.norm = nn.ModuleList()

        for j in range(1, layers + 1):
            
            rf_size_j = int(1 + (ker - 1) * (dilation_exp**j - 1) / (dilation_exp - 1)) if dilation_exp > 1 else 1 + j * (ker - 1)
           
            if self.seq_len>self.receptive_field:
                self.skip_convs.append(nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, self.seq_len-rf_size_j+1)))
            else:
                self.skip_convs.append(nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, self.receptive_field-rf_size_j+1)))

            if self.seq_len>self.receptive_field:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.seq_len - rf_size_j + 1),elementwise_affine=layer_norm_affine))
            else:
                self.norm.append(LayerNorm((residual_channels, num_nodes, self.receptive_field - rf_size_j + 1),elementwise_affine=layer_norm_affine))

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(end_channels, pred_len, kernel_size=(1,1), bias=True)
        
        if self.seq_len > self.receptive_field:
            self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.seq_len), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, self.seq_len-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_dim, skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1), bias=True)

        self.register_buffer("idx", torch.arange(self.num_nodes), persistent=False)

    def forecast(self, input, idx=None):
        if len(input.shape) == 3:
            input = input.unsqueeze(-1)
        input = input.transpose(1, 3)
        seq_len = input.size(3)
        assert seq_len==self.seq_len, 'input sequence length not equal to preset sequence length'

        if self.seq_len<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_len,0,0,0))

        if self.gcn_true:
            if idx is None:
                adp = self.gc(self.idx)
            else:
                adp = self.gc(idx)
           

        x = self.start_conv(input)
        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        for i in range(self.layers):
            residual = x
            filter = self.filter_convs[i](x)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x = self.gconv1[i](x, adp)+self.gconv2[i](x, adp.transpose(1,0))
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x,self.idx)
            else:
                x = self.norm[i](x,idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        x = x.squeeze(-1)
        return x

    def forward(self, input, idx=None):
        history = input[:, -1:]
        output = self.forecast(input - history)
        return output + history
