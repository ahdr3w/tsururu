import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numbers




class NConv(nn.Module):
    def __init__(self):
        super(NConv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncwl,vw->ncvl',(x,A))
        return x.contiguous()


class DyNConv(nn.Module):
    def __init__(self):
        super(DyNConv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,nvwl->ncwl',(x,A))
        return x.contiguous()


class Linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(Linear,self).__init__()

        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=1, bias=bias)

    def forward(self,x):
        return self.mlp(x)


class Prop(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(Prop, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear(c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        dv = d
        a = adj / dv.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
        ho = self.mlp(h)
        return ho


class MixProp(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(MixProp, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear((gdep+1)*c_in,c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha

    def forward(self,x,adj):
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        h = x
        out = [h]
        a = adj / d.view(-1, 1)
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,a)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho = self.mlp(ho)
        return ho


class DyMixProp(nn.Module):
    def __init__(self,c_in,c_out,gdep,dropout,alpha):
        super(DyMixProp, self).__init__()
        self.nconv = DyNConv()
        self.mlp1 = Linear((gdep+1)*c_in,c_out)
        self.mlp2 = Linear((gdep+1)*c_in,c_out)

        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.lin1 = Linear(c_in,c_in)
        self.lin2 = Linear(c_in,c_in)

    def forward(self,x):
        #adj = adj + torch.eye(adj.size(0)).to(x.device)
        #d = adj.sum(1)
        x1 = torch.tanh(self.lin1(x))
        x2 = torch.tanh(self.lin2(x))
        adj = self.nconv(x1.transpose(2,1),x2)
        adj0 = torch.softmax(adj, dim=2)
        adj1 = torch.softmax(adj.transpose(2,1), dim=2)
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha*x + (1-self.alpha)*self.nconv(h,adj0)
            out.append(h)
        ho = torch.cat(out,dim=1)
        ho1 = self.mlp1(ho)
        h = x
        out = [h]
        for i in range(self.gdep):
            h = self.alpha * x + (1 - self.alpha) * self.nconv(h, adj1)
            out.append(h)
        ho = torch.cat(out, dim=1)
        ho2 = self.mlp2(ho)

        return ho1+ho2


class Dilated1D(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(Dilated1D, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        self.tconv = nn.Conv2d(cin,cout,(1,7),dilation=(1,dilation_factor))

    def forward(self,input):
        x = self.tconv(input)
        return x

class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2,3,6,7]
        cout = int(cout/len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin,cout,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][...,-x[-1].size(3):]
        x = torch.cat(x,dim=1)
        return x
    

class LayerNorm(nn.Module):
    __constants__ = ['normalized_shape', 'weight', 'bias', 'eps', 'elementwise_affine']
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input, idx):
        if self.elementwise_affine:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight[:,idx,:], self.bias[:,idx,:], self.eps)
        else:
            return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class SpatialEmbedding(nn.Module):
    def __init__(self, c_in, d_model, num_nodes=7 ,dropout=0.1):
        super(SpatialEmbedding, self).__init__()
        self.d_model = d_model
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

        # spatial embeddings
        self.node_emb = nn.Parameter(torch.empty(num_nodes, self.d_model))
        nn.init.xavier_uniform_(self.node_emb)
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        B, N, T = x.shape
        node_emb = self.node_emb.unsqueeze(0).expand(B, -1, -1)
        # _, _, f = x_mark.shape
        
        x = self.value_embedding(x) + node_emb

        #x: [Batch Variate d_model]
        return self.dropout(x)


class Predict(nn.Module):
    def __init__(self,  individual, c_out, seq_len, pred_len ,dropout):
        super(Predict, self).__init__()
        self.individual = individual
        self.c_out = c_out

        if self.individual:
            self.seq2pred = nn.ModuleList()
            self.dropout = nn.ModuleList()
            for i in range(self.c_out):
                self.seq2pred.append(nn.Linear(seq_len , pred_len))
                self.dropout.append(nn.Dropout(dropout))
        else:
            self.seq2pred = nn.Linear(seq_len , pred_len)
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.individual:
            out = []
            for i in range(self.c_out):
                per_out = self.seq2pred[i](x[:,i,:])
                per_out = self.dropout[i](per_out)
                out.append(per_out)
            out = torch.stack(out,dim=1)
        else:
            out = self.seq2pred(x)
            out = self.dropout(out)

        return out
    

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        # x = self.conv(x)
        return x    
    

class GFC(nn.Module):
    def __init__(self, c_in, c_out, gcn_depth, dropout , seg=4):
        super(GFC, self).__init__()
        self.nconv = NConv()
        self.mlp = Linear(c_in , c_out)
        self.dropout = dropout
        self.alpha = 0.5
        self.seg = seg
        self.seg_dim = c_in // self.seg
        self.pad = c_in % self.seg
        self.agg = nn.ModuleList()
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,3], stride=1, padding=[0,1]))
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,5], stride=1, padding=[0,2]))
        self.agg.append(SeparableConv2d(c_in//seg, c_in//seg, kernel_size=[1,7], stride=1, padding=[0,3]))


    #(B, c, N, d_model)
    def forward(self, x, adj):
        #adj
        adj = adj + torch.eye(adj.size(0)).to(x.device)
        d = adj.sum(1)
        a = adj / d.view(-1, 1)

        #split
        if self.pad == 0:
            x = x.split([self.seg_dim] * self.seg, dim=1)
            out = [x[0]]
            # (B, c, N ,d_model)
            for i in range(1, self.seg):
                h = self.agg[i-1](x[i])
                h = self.alpha * (h + x[i]) + (1 - self.alpha) * self.nconv(h, a)
                out.append(h)
        else:
            y = x[:, :self.seg_dim + self.pad, :, :]
            out = [y]
            x = x[:, self.seg_dim + self.pad:, :, :]
            x = x.split([self.seg_dim] * (self.seg-1),dim=1)
            # (B, c, N ,d_model)
            for i in range(0, self.seg-1):
                h = self.agg[i](x[i])
                h = self.alpha * (h + x[i]) + (1 - self.alpha) * self.nconv(h, a)
                out.append(h)

        out = torch.cat(out,dim=1)
        out = self.mlp(out)
        return  out
    



