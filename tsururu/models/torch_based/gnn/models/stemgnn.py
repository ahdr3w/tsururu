import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph import AdjacencyMatrix


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi,
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        ffted = torch.fft.fft(input, dim=1)
        real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        iffted = torch.fft.ifft(torch.complex(real, img), dim=1).real
        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x)
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


class Model(nn.Module):
    """StemGNN model from the paper [Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting](https://arxiv.org/pdf/2103.07719).

    Args:
        model_params (dict): Dictionary containing parameters for the model:
            - seq_len (int): Temporal length of time series.
            - pred_len (int): Length of prediction.
            - num_nodes (int): Number of nodes/variables to be forecasted.
            - multi_layer (int): Hyper parameter of STemGNN which controls the parameter number of hidden layers.
            - stack_cnt (int): Number of blocks.
            - dropout_rate (float): Dropout ratio.
            - leaky_rate (float): Leaky Relu ratio. 
    """

    def __init__(self, seq_len, pred_len, num_nodes, multi_layer, stack_cnt, dropout_rate=0.5, leaky_rate=0.2):
        super(Model, self).__init__()
        
        self.stack_cnt = stack_cnt
        
        self.adj = AdjacencyMatrix(strategy='dynamic_attention', seq_len=seq_len, num_nodes=num_nodes, dropout_rate=dropout_rate, 
                                   leaky_rate=leaky_rate)
        
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [StockBlockLayer(seq_len, num_nodes, multi_layer, stack_cnt=i) for i in range(stack_cnt)])
        self.fc = nn.Sequential(
            nn.Linear(int(seq_len), int(seq_len)),
            nn.LeakyReLU(),
            nn.Linear(int(seq_len), pred_len),
        )

        self.leakyrelu = nn.LeakyReLU(leaky_rate)
        self.dropout = nn.Dropout(p=dropout_rate)
       

    @staticmethod
    def get_laplacian(adj):
        """
        return the laplacian of the graph.
        
        """
        degree = torch.sum(adj, dim=1)
        # laplacian is sym or not
        adj = 0.5 * (adj + adj.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - adj, diagonal_degree_hat))
        return laplacian

    @staticmethod
    def cheb_polynomial(laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        adj = self.adj(x)
        
        laplacian = self.get_laplacian(adj)
        mul_L = self.cheb_polynomial(laplacian)

        return mul_L

    def forward(self, x):
        mul_L = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        if forecast.size()[-1] == 1:
            return forecast.unsqueeze(1).squeeze(-1)
        else:
            return forecast.permute(0, 2, 1).contiguous()