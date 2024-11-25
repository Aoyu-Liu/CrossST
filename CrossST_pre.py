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
        # torch.Size([64, 12, 211, 3])
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
        # torch.Size([64, 211, 12, 128])
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
    def __init__(self, device, d_model, head, seq_length=1, dropout=0.1):
        super(SpatialAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head 
        self.head = head
        self.seq_length = seq_length
        self.d_model = d_model
        self.q = Conv(d_model)
        self.v = Conv(d_model)
        self.concat = Conv(d_model)

        self.s_bank1 = nn.Parameter(torch.randn(head, seq_length, 2500, self.d_k))
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
        return x


class SEncoder(nn.Module):
    def __init__(self, device, d_model, head, seq_length=1, dropout=0.1):
        "Take in model size and number of heads."
        super(SEncoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head  
        self.head = head
        self.seq_length = seq_length
        self.d_model = d_model

        self.s_bank1 = nn.Parameter(torch.ones(d_model, 2500, seq_length))
        self.s_bank1_bias = nn.Parameter(torch.zeros(d_model, 2500, seq_length))

        self.s_bank2 = nn.Parameter(torch.ones(d_model, 2500, seq_length))
        self.s_bank2_bias = nn.Parameter(torch.zeros(d_model, 2500, seq_length))

        self.spa_attn = SpatialAttention(
            device, d_model, head, seq_length=seq_length
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
        mode = "pre-train"
    ):
        super().__init__()

        # attributes
        self.device = device
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.mode = mode
        self.head = 1


        if input_len==96:
            patch = 4
        elif input_len==12:
            patch = 1 
        elif input_len==288:
            patch = 12

        self.patch_embedding = PatchEmbedding(
            channels, patch_len=patch, stride=patch,  his=input_len)

        self.tencoder = TModule(input_dim, channels, input_len, self.input_len//patch)

        self.sencoder = SEncoder(
            device,
            d_model=channels,
            head=self.head,
            seq_length=1,
            dropout=dropout,
        )

        self.regression_layer = nn.Conv2d(
            channels, self.output_len, kernel_size=(1, 1)
        )

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def forward(self, x):
        # b, t, n, c
        batch, time_steps, num_nodes, channel = x.shape
        h = self.patch_embedding(x)
        h_t = self.tencoder(h)
        h_s = self.sencoder(h_t)
        y = self.regression_layer(h_s)
        if self.mode == "pre-train":
            y = self.regression_layer(h_s)
            y = y[:, :time_steps, :, :]
            return y
        else:
            return h, h_t, h_s