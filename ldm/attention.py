import torch
from torch import nn
from einops import rearrange
import math

class SelfAttention(nn.Module):
    def __init__(self, ni, attn_chans):
        super().__init__()
        self.attn = nn.MultiheadAttention(ni, ni//attn_chans, batch_first=True)
        self.norm = nn.BatchNorm2d(ni)

    def forward(self, x):
        n, c, h, w = x.shape
        x_norm = self.norm(x.clone()) 
        x_norm = x_norm.view(n, c, -1).transpose(1, 2)
        x = self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        return x.transpose(1,2).reshape(n,c,h,w)

# class SelfAttention(nn.Module):
#     def __init__(self, ni, attn_chans):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(ni, ni//attn_chans, batch_first=True)
#         self.norm = nn.BatchNorm2d(ni)

#     def forward(self, x):
#         n,c,h,w = x.shape
#         x = self.norm(x).view(n, c, -1).transpose(1, 2)
#         x = self.attn(x, x, x, need_weights=False)[0]
#         return x.transpose(1,2).reshape(n,c,h,w)

# class SelfAttentionMH(nn.Module):
#     def __init__(self, ni, attn_chans, transpose=True):
#         super().__init__()
#         self.nheads = ni//attn_chans
#         self.scale = math.sqrt(ni/self.nheads)
#         self.norm = nn.LayerNorm(ni)
#         self.qkv = nn.Linear(ni, ni*3)
#         self.proj = nn.Linear(ni, ni)
#         self.t = transpose
    
#     def forward(self, x):
#         n,c,s = x.shape
#         if self.t: x = x.transpose(1, 2)
#         x = self.norm(x)
#         x = self.qkv(x)
#         x = rearrange(x, 'n s (h d) -> (n h) s d', h=self.nheads)
#         q,k,v = torch.chunk(x, 3, dim=-1)
#         s = (q@k.transpose(1,2))/self.scale
#         x = s.softmax(dim=-1)@v
#         x = rearrange(x, '(n h) s d -> n s (h d)', h=self.nheads)
#         x = self.proj(x)
#         if self.t: x = x.transpose(1, 2)
#         return x

# class SelfAttention(SelfAttentionMH):
#     def forward(self, x):
#         n,c,h,w = x.shape
#         return super().forward(x.view(n, c, -1)).reshape(n,c,h,w)