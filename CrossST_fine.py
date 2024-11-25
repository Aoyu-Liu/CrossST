from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        return input


class PatchEmbedding(nn.Module):
    def __init__(self, d_model=128, patch_len=1, stride=1, his=12):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.his = his
        self.value_embedding = nn.Linear(self.patch_len*3, d_model, bias=False)
        self.tem_embedding = torch.nn.Parameter(torch.zeros(1, 1, his, d_model))

    def forward(self, x):
        batch, _, num_nodes, _ = x.size()
        x = x.permute(0, 2, 3, 1)
        if self.his == x.shape[-1]:
            x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            x = x.transpose(2, 3).contiguous().view(batch, num_nodes, self.his//self.patch_len, -1)
        else:
            gap = self.his // x.shape[-1]
            x = x.unfold(dimension=-1, size=self.patch_len//gap, step=self.stride//gap)
            x = x.transpose(2, 3).contiguous().view(batch, num_nodes, self.his//self.patch_len, -1)
            x = F.pad(x, (0, (self.patch_len - self.patch_len//gap)))
        x = self.value_embedding(x) + self.tem_embedding[:, :, :x.size(2), :]
        return x 
        

class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SpatialAttention(nn.Module): 
    def __init__(self, device, d_model, num_nodes, head, seq_length=1, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head 
        self.head = head
        self.seq_length = seq_length
        self.d_model = d_model
        self.q = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

        self.s_bank1 = nn.Parameter(torch.randn(head, seq_length, num_nodes, self.d_k))
        nn.init.xavier_uniform_(self.s_bank1)


    def forward(self, input, adj_list=None):
        batch, channel, num_nodes, time_step = input.shape
        query, value = self.q(input), self.v(input)
        query = query.view(
            query.shape[0], -1, self.d_k, query.shape[2], self.seq_length
        ).permute(0, 1, 4, 3, 2)
        value = value.view(
            value.shape[0], -1, self.d_k, value.shape[2], self.seq_length
        ).permute(
            0, 1, 4, 3, 2
        )  
        key = torch.softmax(self.s_bank1[:,:,:num_nodes,:] / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)
        kv = torch.einsum("hlnx, bhlny->bhlxy", key, value)
        attn_qkv = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)
        x = attn_qkv
        x = (
            x.permute(0, 1, 4, 3, 2)
            .contiguous()
            .view(x.shape[0], self.d_model, num_nodes, self.seq_length)
        )
        x = self.concat(x)
        s_bank = self.s_bank1.squeeze()[:num_nodes,].permute(1,0).unsqueeze(-1)
        out = x*s_bank
        return out, s_bank


class SEncoder(nn.Module):
    def __init__(self, device, d_model,  num_nodes, head, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(SEncoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.seq_length = seq_length
        self.d_model = d_model

        self.s_bank1 = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))
        self.s_bank1_bias = nn.Parameter(torch.zeros(d_model, num_nodes, seq_length))

        self.s_bank2 = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))
        self.s_bank2_bias = nn.Parameter(torch.zeros(d_model, num_nodes, seq_length))

        self.spa_attn = SpatialAttention(
            device, d_model, num_nodes, head, seq_length=seq_length
        )

        self.glu = GLU(d_model)

        self.fc = nn.Conv2d(
            d_model, d_model, kernel_size=(1, 1)
        )

        self.norm1 = LayerNorm()  
        self.norm2 = LayerNorm()  

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, h):
        # Self-Attention
        b, d, n, l = h.shape
        h_s = h + self.dropout1(self.spa_attn(h))
        h_s = self.norm1(h_s)*self.s_bank1[:,:n,:] +self.s_bank1_bias[:,:n,:]

        # Feedforward Neural Network
        h_f = h_s + self.dropout2(self.glu(h_s))
        h_f = self.norm2(h_f)*self.s_bank2[:,:n,:] +self.s_bank2_bias[:,:n,:]
        return h_f + self.fc(h)


class TConv(nn.Module):
    def __init__(self, features=128, layer=4, length=12, dropout=0.1):
        super(TConv, self).__init__()
        layers = []
        kernel_size = int(length / layer + 1)
        for i in range(layer):
            self.conv = nn.Conv2d(features, features, (1, kernel_size))
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            layers += [nn.Sequential(self.conv, self.relu, self.dropout)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.tcn(x) + x[..., -1].unsqueeze(-1)
        return x


class GlobalFilter(nn.Module):
    def __init__(self, dim=3, channels=128, time_step=96, num_patches=24):
        super().__init__()
        self.time_step = time_step
        self.num_patches = num_patches
        self.complex_weight = nn.Parameter(torch.randn(channels, 1, num_patches//2 + 1, 2, dtype=torch.float32) * 0.02)
        nn.init.xavier_uniform_(self.complex_weight)
        self.conv = nn.Conv2d(channels, channels,(1,1))
        self.fc = nn.Linear(num_patches, 1)

    def forward(self, x):
        B, C, N, _ = x.shape
        x = torch.fft.rfft(x, dim=3, norm='ortho')
        weight= torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft(x, n=self.num_patches, dim=3, norm='ortho')
        return x


class TModule(nn.Module):
    def __init__(self, input_dim = 3, channels = 128, time_step=96, num_patches=24):
        "Take in model size and number of heads."
        super(TModule, self).__init__()
        self.fft = GlobalFilter(input_dim, channels, time_step, num_patches)
        self.tconv = TConv(channels, layer=3, length=num_patches)

    def forward(self, h):
        h = h.permute(0,3,1,2)
        h1 = self.fft(h)    
        h2 = self.tconv(h1)
        return h2


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim,  out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)

    def forward(self, input_data):
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data)))) 
        hidden = hidden + input_data                          
        return hidden

        
class STAMT(nn.Module):
    def __init__(
        self,
        device,
        input_dim=3,
        channels=64,
        num_nodes=170,
        input_len=12,
        output_len=12,
        dropout=0.1,
        t_model = None
    ):
        super().__init__()
        # attributes
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1
        
        self.t_patch_embedding = t_model.patch_embedding
        self.t_tencoder = t_model.tencoder
        self.t_sencoder = t_model.sencoder

        if input_len==96:
            patch = 4
        elif input_len==12:
            patch = 1 
        elif input_len==288:
            patch = 12

        s_channels = 64

        self.patch_embedding = PatchEmbedding(
            s_channels, patch_len=patch, stride=patch,  his=input_len)

        self.tencoder = TModule(input_dim, s_channels, input_len, self.input_len//patch)

        self.sencoder = SEncoder(
            device,
            d_model=s_channels,
            num_nodes = num_nodes,
            head=self.head,
            seq_length=1,
            dropout=dropout,
        )

        self.fc_t = nn.Sequential(nn.Conv2d(s_channels, channels, kernel_size=(1,1), stride=(1,1)),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))

        self.fc_s = nn.Sequential(nn.Conv2d(s_channels, channels, kernel_size=(1,1), stride=(1,1)),
                                    nn.ReLU(),
                                    nn.Dropout(dropout))
        
        self.mask_t = nn.Parameter(torch.ones([channels, num_nodes, 1]))
        self.bias_t = nn.Parameter(torch.zeros([channels, num_nodes, 1]))

        self.mask_s = nn.Parameter(torch.ones([channels, num_nodes, 1]))
        self.bias_s = nn.Parameter(torch.zeros([channels, num_nodes, 1]))

        self.mlp = MultiLayerPerceptron(channels*2, channels*2)    

        self.regression_layer = nn.Conv2d(
            channels*2, self.output_len, kernel_size=(1, 1)
        )


    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, input):
        
        batch, time_steps, num_nodes, num_feas = input.size()
        
        with torch.no_grad():
            t_h = self.t_patch_embedding(input)
            t_h_t = self.t_tencoder(t_h)
            t_h_s = self.t_sencoder(t_h_t)

        h = self.patch_embedding(input)
        h_t = self.tencoder(h)
        h_s = self.sencoder(h_t)

        h_t = self.fc_t(h_t)
        h_s = self.fc_s(h_s)

        st_hidden = torch.cat([h_t+ (self.mask_t*t_h_t + self.bias_t)] + [h_s + (self.mask_s*t_h_s + self.bias_s)], dim=1)
        
        st_hidden = self.mlp(st_hidden)

        y = self.regression_layer(st_hidden)
        y = y[:, :time_steps, :, :]
        return y, t_h_t, t_h_s, h_t, h_s
        