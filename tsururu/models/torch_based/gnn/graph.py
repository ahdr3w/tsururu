import torch
import torch.nn.functional as F
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, n_nodes, **kwargs):
        super(Identity, self).__init__()
        self.n_nodes = n_nodes
        self.A = nn.Parameter(torch.eye(n_nodes), requires_grad=False)

    def forward(self, idx):
        return self.A


class GlobalLinear(nn.Module):
    def __init__(self, n_nodes, **kwargs):
        super(GlobalLinear, self).__init__()
        self.n_nodes = n_nodes
        self.A = nn.Parameter(torch.randn(n_nodes, n_nodes), requires_grad=True)

    def forward(self, idx):
        return self.A

class Global(nn.Module):
    def __init__(self, n_nodes, **kwargs):
        super(Global, self).__init__()
        self.n_nodes = n_nodes
        self.A = nn.Parameter(torch.randn(n_nodes, n_nodes), requires_grad=True)

    def forward(self, idx):
        return F.relu(self.A)
    

class Directed(nn.Module):
    def __init__(self, n_nodes, k, dim, alpha=3, static_feat=None, **kwargs):
        super(Directed, self).__init__()
        self.n_nodes = n_nodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(n_nodes, dim)
            self.emb2 = nn.Embedding(n_nodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, x):
        if self.static_feat is None:
            nodevec1 = self.emb1(x)
            nodevec2 = self.emb2(x)
        else:
            nodevec1 = self.static_feat[x,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(x.size(0), x.size(0)).to(x.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class Undirected(nn.Module):
    def __init__(self, n_nodes, k, dim, alpha=3, static_feat=None, **kwargs):
        super(Undirected, self).__init__()
        self.n_nodes = n_nodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(n_nodes, dim)
            self.lin1 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, x):
        if self.static_feat is None:
            nodevec1 = self.emb1(x)
            nodevec2 = self.emb1(x)
        else:
            nodevec1 = self.static_feat[x,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin1(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(x.size(0), x.size(0)).to(x.device)
        mask.fill_(float('0'))
        s1,t1 = adj.topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class Unidirected(nn.Module):
    def __init__(self, n_nodes, k, dim, alpha=3, **kwargs):
        super(Unidirected, self).__init__()
        self.n_nodes = n_nodes
      
        self.emb1 = nn.Embedding(n_nodes, dim)
        self.emb2 = nn.Embedding(n_nodes, dim)
        self.lin1 = nn.Linear(dim,dim)
        self.lin2 = nn.Linear(dim,dim)

        self.k = k
        self.dim = dim
        self.alpha = alpha

    def forward(self, x):
        
        nodevec1 = self.emb1(x)
        nodevec2 = self.emb2(x)
        

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(x.size(0), x.size(0)).to(x.device)
        mask.fill_(float('0'))
        s1,t1 = (adj + torch.rand_like(adj)*0.01).topk(self.k,1)
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj


class GraphWaveNET(nn.Module):
    def __init__(self, n_nodes, dim, **kwargs):
        super(GraphWaveNET, self).__init__()

        self.nodevec1 = nn.Parameter(torch.randn(n_nodes, dim), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(dim, n_nodes), requires_grad=True)
        

    def forward(self, x):
        adj = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)

        return adj 


class DynamicAttention(nn.Module):
    def __init__(self, attn_len, n_nodes, dropout_rate=0.5, leaky_rate=0.2, **kwargs):
        super(DynamicAttention, self).__init__()
       
        self.weight_key = nn.Parameter(torch.zeros(size=(n_nodes, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(n_nodes, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)

        self.gru = nn.GRU(attn_len, n_nodes)
        
        self.leakyrelu = nn.LeakyReLU(leaky_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention
    
    def forward(self, x):
        input, _ = self.gru(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        adj = self.self_graph_attention(input)
        adj = torch.mean(adj, dim=0)
        
        return adj

class MultiheadAttention(nn.Module):
    def __init__(self, attn_len, num_heads, **kwargs):
        super(MultiheadAttention, self).__init__()

        self.attn = nn.MultiheadAttention(attn_len, num_heads, dropout=0.0, batch_first=True)

        self.query = nn.Linear(attn_len, attn_len)
        self.key = nn.Linear(attn_len, attn_len)
        self.value = nn.Linear(attn_len, attn_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        _, adj = self.attn(q, k, v)

        adj = adj.mean(dim=0)

        return adj



    
class AdjacencyMatrix(nn.Module):
    """This class introduces data-driven approaches to build Adjacency Matrix,
    representing graph of time series.  

    Args:
        strategy: specify what strategy of strategies list to use.
        List["global", "directed", "undirected", "unidirected", "graph_wavenet", "dynamic_attention"]

        **kwargs: arguments that passed to a strategy

    Notes:
        There are two types of strategies available by now. 

            1) In first category, created graph is static and doesn't require time series as input; it needs 1d integer tensor 
            as input for forward function, it represents nodes that should be considered to build a subgraph.

            2) In second category, created graph is dynamic and, thus, it requires time series as input.


    """

    strategies = {
        "identity": Identity,
        "linear" : GlobalLinear,
        "global" : Global,
        "directed" : Directed,
        "undirected" : Undirected,
        "unidirected" : Unidirected,
        "graph_wavenet" : GraphWaveNET,
        "dynamic_attention" : DynamicAttention,
        "multihead_attention" : MultiheadAttention
    }

    def __init__(self, adj="global", **kwargs):
        super(AdjacencyMatrix, self).__init__()
        self.adj_type = adj
        self.adjacency_matrix = self.strategies[adj](**kwargs)

    def forward(self, x):
        if "dynamic_attention" in self.adj_type or "multihead_attention" in self.adj_type:
            return self.adjacency_matrix(x)
        elif len(x.shape) == 1:
            return self.adjacency_matrix(x)
        else:
            idx = torch.arange(x.shape[2]).to(x.device)
            return self.adjacency_matrix(idx)
    
    def __repr__(self):
        output = "AdjacencyMatrix\n"
        output += f"Strategies to build a graph: {self.strategies.keys()}\n"
        if self.adjacency_matrix is not None:
            output += "Current:\n\t"
            output += self.adjacency_matrix.__repr__().replace("\n", "\n\t")
        return output 
