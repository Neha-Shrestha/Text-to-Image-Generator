import torch
from torch import nn
import torch.nn.functional as F
import math

class TimestepEmbedding(nn.Module):
    def __init__(self, n_channels):
        self.n_channels = n_channels
        self.n_embedding = self.n_channels * 4
        self.timestep_mlp = nn.Sequential(
            nn.Linear(self.n_channels, self.n_embedding),
            nn.SiLU(),
            nn.Linear(self.n_embedding, self.n_embedding)
        )

    def forward(self, t):
        emb = -math.log(10000) * torch.linspace(0, 1, self.n_embedding//2)
        emb = t[:, None] * emb.exp()[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = self.timestep_mlp(emb)
        return emb

# def timestep_embedding(tsteps, emb_dim, max_period= 10000):
#     exponent = -math.log(max_period) * torch.linspace(0, 1, emb_dim//2, device=tsteps.device)
#     emb = tsteps[:,None].float() * exponent.exp()[None,:]
#     emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
#     return F.pad(emb, (0,1,0,0)) if emb_dim%2==1 else emb