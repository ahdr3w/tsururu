import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange
import numpy as np

from layers import EncoderLayer
from layers import FullAttention, AttentionLayer
from layers import PatchEmbedding

from ..graph import AdjacencyMatrix


torch.set_printoptions(profile='short', linewidth=200)

class Flatten_Head(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('nwl,vw->nvl',(x,A))
        return x.contiguous()


class mixprop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout=0.2,alpha=0.1):
        super(mixprop, self).__init__()
        self.nconv = nconv()
        self.mlp = nn.Linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        # normalization to adj
        d = adj.sum(1)
        a = adj / d.view(-1, 1)
        h = x
        out = [h]
        for _ in range(self.gdep):
            # h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            h = F.dropout(h, self.dropout)
            h = self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=2)
        ho = self.mlp(ho)
        return ho


class GraphEncoder(nn.Module):
    def __init__(self, attn_layers, gnn_layers, gl_layer, cls_len, norm_layer=None):
        super(GraphEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.graph_layers = nn.ModuleList(gnn_layers)
        self.graph_learning = gl_layer
        self.norm = norm_layer
        self.cls_len = cls_len

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        gcls_len = self.cls_len
        nodes = torch.arange(self.graph_learning.n_nodes).to(x.device)
        adj = self.graph_learning(nodes)

        for i, attn_layer in enumerate(self.attn_layers):
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

            if i < len(self.graph_layers):
                g = x[:,:gcls_len]
                g = rearrange(g, '(b n) p d -> (b p) n d', n=self.node_embs.shape[0])
                g = self.graph_layers[i](g, adj) + g
                g = rearrange(g, '(b p) n d -> (b n) p d', p=gcls_len)
                x[:,:gcls_len] = g

            if self.norm is not None:
                x = self.norm(x)

        return x, attns

class Model(nn.Module):
    """SageFormer model from the paper [SageFormer: Series-Aware Framework for Long-term Multivariate Time Series Forecasting](https://arxiv.org/pdf/2211.14730.pdf).

    Args:
        model_params (dict): Dictionary containing parameters for the model:
            - seq_len (int): Temporal length of time series.
            - pred_len (int): Length of prediction.
            - enc_in (int): Number of nodes/variables to be forecasted.
            - patch_len (int): Patch len for patch embedding.
            - stride (int): Stride for patch embedding.
            - cls_len (int): numer of cls tokens
    """

    def __init__(self, seq_len, pred_len, enc_in, patch_len=16, stride=8, cls_len=3, 
                 gdep=3, knn=16, embed_dim=16, d_model=512, n_heads=8, d_ff=2048, e_layers=2,  
                 gc_alpha=1, dropout=0.1, factor=1, output_attention=False, activation="gelu"):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        padding = stride
        
        

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, padding, dropout)
        # global tokens
        self.cls_token = nn.Parameter(torch.randn(1, cls_len, d_model))
        # Encoder
        self.encoder = GraphEncoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [mixprop(d_model, d_model, gdep) for _ in range(e_layers-1)],
            

            AdjacencyMatrix("undirected", enc_in, knn, embed_dim, gc_alpha),
            cls_len,
            norm_layer=nn.LayerNorm(d_model)
        )

        # Prediction Head
        self.head_nf = d_model * \
            int((seq_len - patch_len) / stride + 2)
        
        self.head = Flatten_Head(enc_in, self.head_nf, pred_len,
                                    head_dropout=dropout)
        

    def forward(self, x):
        # Normalization from Non-stationary Transformer
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        # do patching and embedding
        x = x.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x)
        # cls token
        patch_len = enc_out.shape[1]
        cls_tokens = self.cls_token.repeat(enc_out.shape[0], 1, 1)
        enc_out = torch.cat([cls_tokens, enc_out], dim=1)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        enc_out = enc_out[:,-patch_len:,:]
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    