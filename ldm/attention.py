import torch
from torch import nn
from einops import rearrange
import math

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, ic, nheads, transpose=True):
        super().__init__()
        self.nheads = ic//nheads
        self.scale = math.sqrt(ic/self.nheads)
        self.norm = nn.LayerNorm(ic)
        self.qkv = nn.Linear(ic, ic*3)
        self.proj = nn.Linear(ic, ic)
        self.t = transpose
    
    def forward(self, inp):
        n,c,h,w = inp.shape
        if self.t: x = x.transpose(1, 2)
        x = self.norm(inp).view(n, c, -1).transpose(1, 2)
        x = self.qkv(x)
        x = rearrange(x, 'n s (h d) -> (n h) s d', h=self.nheads)
        q,k,v = torch.chunk(x, 3, dim=-1)
        s = (q@k.transpose(1,2))/self.scale
        x = s.softmax(dim=-1) @ v
        x = rearrange(x, '(n h) s d -> n s (h d)', h=self.nheads)
        x = self.proj(x)
        if self.t: x = x.transpose(1,2).reshape(n,c,h,w)
        return x