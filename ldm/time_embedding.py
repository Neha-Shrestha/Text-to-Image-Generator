import torch
from torch import nn
import torch.nn.functional as F
import math

class TimestepEmbedding(nn.Module):
    def __init__(self, n_channels, n_embedding):
        super().__init__()
        self.n_channels = n_channels
        self.n_embedding = n_embedding
        self.timestep_mlp = nn.Sequential(
            nn.BatchNorm1d(self.n_channels),
            nn.SiLU(),
            nn.Linear(self.n_channels, self.n_embedding),
            nn.SiLU(),
            nn.Linear(self.n_embedding, self.n_embedding)
        )

    def forward(self, t):
        exponent = -math.log(10000) * torch.linspace(0, 1, self.n_channels//2, device=t.device)
        emb = t[:, None].float() * exponent.exp()[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        emb = self.timestep_mlp(emb)
        return emb