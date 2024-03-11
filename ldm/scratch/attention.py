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
