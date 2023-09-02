import torch
import torch.nn as nn
from einops import rearrange
from .transformer import PositionalEncoding

class Attention(nn.Module):
    def __init__(self, in_channels=512, max_length=25, n_feature=256):
        super().__init__()
        self.max_length = max_length

        self.f0_embedding = nn.Embedding(max_length, in_channels)
        self.w0 = nn.Linear(max_length, n_feature)
        self.wv = nn.Linear(in_channels, in_channels)
        self.we = nn.Linear(in_channels, max_length)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        enc_output = enc_output.permute(0, 2, 3, 1).flatten(1, 2)
        reading_order = torch.arange(self.max_length, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)  # (S,) -> (B, S)
        reading_order_embed = self.f0_embedding(reading_order)  # b,25,512

        t = self.w0(reading_order_embed.permute(0, 2, 1))  # b,512,256
        t = self.active(t.permute(0, 2, 1) + self.wv(enc_output))  # b,256,512

        attn = self.we(t)  # b,256,25
        attn = self.softmax(attn.permute(0, 2, 1))  # b,25,256
        g_output = torch.bmm(attn, enc_output)  # b,25,512
        return g_output, attn.view(*attn.shape[:2], 8, 32)

class EncoderLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(True)
    def forward(self, x, output_size):
        x = self.conv(x, output_size=output_size)
        x = self.bn(x)
        x = self.relu(x)
        return x

class PositionAttention(nn.Module):
    def __init__(self, max_length, in_channels=512, num_channels=64):
        super().__init__()
        self.max_length = max_length
        self.k_encoder = nn.Sequential(
            EncoderLayer(in_channels , num_channels, stride=(1, 2)),
            EncoderLayer(num_channels, num_channels, stride=(2, 2)),
            EncoderLayer(num_channels, num_channels, stride=(2, 2)),
            EncoderLayer(num_channels, num_channels, stride=(2, 2)),
        )
        self.k_decoder = nn.Sequential(
            DecoderLayer(num_channels, num_channels, stride=(2, 2)),
            DecoderLayer(num_channels, num_channels, stride=(2, 2)),
            DecoderLayer(num_channels, num_channels, stride=(2, 2)),
            DecoderLayer(num_channels, in_channels,  stride=(1, 2)),
        )
        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)

    def forward(self, x):
        B, C, H, W = x.size()
        k, v = x, x  # (B, C, H, W)
        # Calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k, output_size=features[len(self.k_decoder) - 2 - i].shape)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k, output_size=v.shape)
        k = rearrange(k, 'b c h w -> b c (w h)')

        # Calculate query vector
        zeros = x.new_zeros((self.max_length, B, C))  # (L, B, C)
        q = self.pos_encoder(zeros)  # (L, B, C)
        q = rearrange(q, 'l b c -> b l c')  # (B, L, C)
        q = self.project(q)  # (B, L, C) -> (B, L ,C)
        
        # calculate attention
        attn_scores = torch.bmm(q, k)  # (B, L, C) x (B, C, (W*H)) -> (B, L, (W*H))
        attn_scores = attn_scores / (C ** 0.5)
        attn_scores = torch.softmax(attn_scores, dim=-1)

        v = rearrange(v,'b c h w -> b (w h) c')
        attn_vecs = torch.bmm(attn_scores, v)  # (B, L, (W*H)) x (B, (W*H), C) -> (B, L, C)
        return attn_vecs, rearrange(attn_scores,'b l (w h) -> b l h w', h=H, w=W)
