import torch
import torch.nn.functional as F
from torch import nn

def lin(ic, oc, act=nn.ReLU):
    layers = nn.Sequential(nn.Linear(ic, oc))
    if act: layers.append(act())
    return layers

def vae_loss(X_new=None, X=None, mu=None, lv=None):
    return F.binary_cross_entropy(X_new, X, reduction="sum") + (-0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()))

class VAE(nn.Module):
    def __init__(self, ic, hc, lc):
        super().__init__()
        self.encoder = nn.Sequential(lin(ic, hc), lin(hc, hc))
        self.mu, self.lv = lin(hc, lc, act=None), lin(hc, lc, act=None)
        self.decoder = nn.Sequential(lin(lc, hc), lin(hc,hc), lin(hc, ic, act=None), nn.Sigmoid())
    
    def forward(self, x):
        x = self.encoder(x)
        mu, lv = self.mu(x), self.lv(x)
        ep = torch.randn_like(lv)
        z = mu + ((0.5*lv).exp() * ep)
        return self.decoder(z), mu, lv
