import torch
from torch import nn

def conv(ic, oc, ks=3, s=2, act=True):
    res = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=ks, stride=s, padding=ks//2)
    if act: return nn.Sequential(res, nn.ReLU())
    return res