import torch
import torch.nn as nn
import torch.nn.functional as F




class FGC(nn.Module):
    def __init__(self, scale, frequency_size, hidden_size_factor, sparsity_threshold):
        super(FGC, self).__init__()
        self.sparsity_threshold = sparsity_threshold 

        self.w1 = nn.Parameter(
            scale * torch.randn(2, frequency_size, frequency_size * hidden_size_factor))
        self.b1 = nn.Parameter(scale * torch.randn(2, frequency_size * hidden_size_factor))
        self.w2 = nn.Parameter(
            scale * torch.randn(2, frequency_size * hidden_size_factor, frequency_size))
        self.b2 = nn.Parameter(scale * torch.randn(2, frequency_size))
        self.w3 = nn.Parameter(
            scale * torch.randn(2, frequency_size,
                                     frequency_size * hidden_size_factor))
        self.b3 = nn.Parameter(
            scale * torch.randn(2, frequency_size * hidden_size_factor))

    @staticmethod
    def fourier_graph_operator(x, w, b):
        o_real = F.relu(
            torch.einsum('bli,ii->bli', x[..., 0], w[0]) - \
            torch.einsum('bli,ii->bli', x[..., 1], w[1]) + \
            b[0]
        )

        o_imag = F.relu(
            torch.einsum('bli,ii->bli', x[..., 1], w[0]) + \
            torch.einsum('bli,ii->bli', x[..., 0], w[1]) + \
            b[1]
        )
        out = torch.stack([o_real, o_imag], dim=-1)
        
        return out

    def forward(self, input):
        input = torch.stack([input.real, input.imag], dim=-1)
        
        o1 = self.fourier_graph_operator(input, self.w1, self.b1)
        x = F.softshrink(o1, lambd=self.sparsity_threshold)

        o2 = self.fourier_graph_operator(o1, self.w2, self.b2)
        y = F.softshrink(o2, lambd=self.sparsity_threshold)
        
        o3 = self.fourier_graph_operator(o2, self.w3, self.b3)
        z = F.softshrink(o3, lambd=self.sparsity_threshold)
        
        output = torch.view_as_complex(x + y + z)

        return output


class Model(nn.Module):
    """FourierGNN model from the paper [FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective](https://arxiv.org/pdf/2311.06190).

    Args:
        model_params (dict): Dictionary containing parameters for the model:
            - seq_len (int): Temporal length of time series.
            - pred_len (int): Length of prediction.
            - num_nodes (int): Number of nodes/variables to be forecasted.
            - embed_size (int): Size of embeddings.
            - hidden_size (int): Hidden dimensionality.
    """

    def __init__(self, seq_len, pred_len, embed_size,  hidden_size):
        super().__init__()
        
        number_frequency = 1
        scale = 0.02
        
        self.embed_size = embed_size
        self.frequency_size = self.embed_size // number_frequency
        self.hidden_size_factor = 1
        self.sparsity_threshold = 0.01

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))

        self.fgc = FGC(scale, self.frequency_size, self.hidden_size_factor, self.sparsity_threshold)
        
        self.embeddings_10 = nn.Parameter(torch.randn(seq_len, 8))
        self.fc = nn.Sequential(
            nn.Linear(self.embed_size * 8, 64),
            nn.LeakyReLU(),
            nn.Linear(64, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, pred_len)
        )
        

    def tokenEmb(self, x):
        x = x.unsqueeze(2)
        y = self.embeddings
        return x * y

    def forward(self, x):
        x = x.permute(0, 2, 1).contiguous()
        B, N, L = x.shape
        # B*N*L ==> B*NL
        x = x.reshape(B, -1)
        # embedding B*NL ==> B*NL*D
        x = self.tokenEmb(x)

        # FFT B*NL*D ==> B*NT/2*D
        x = torch.fft.rfft(x, dim=1, norm='ortho')

        x = x.reshape(B, (N*L)//2+1, self.frequency_size)

        bias = x

        # FourierGNN
        x = self.fgc(x)

        x = x + bias

        x = x.reshape(B, (N*L)//2+1, self.embed_size)

        # ifft
        x = torch.fft.irfft(x, n=N*L, dim=1, norm="ortho")

        x = x.reshape(B, N, L, self.embed_size)
        x = x.permute(0, 1, 3, 2)  # B, N, D, L

        # projection
        x = torch.matmul(x, self.embeddings_10)
        x = x.reshape(B, N, -1)
        x = self.fc(x)

        return x.transpose(2, 1)
