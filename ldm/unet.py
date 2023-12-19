import torch
from torch import nn
import math

from time_embedding import *
from attention import *

def unet_conv(ic, oc, ks=3, stride=1, act=nn.SiLU, norm=None, bias=True):
    layers = nn.Sequential()
    if norm: layers.append(norm(ic))
    if act : layers.append(act())
    layers.append(nn.Conv2d(ic, oc, stride=stride, kernel_size=ks, padding=ks//2, bias=bias))
    return layers

def lin(ic, oc, stride=1, act=nn.SiLU, norm=None, bias=True):
    layers = nn.Sequential()
    if norm: layers.append(norm(ic))
    if act : layers.append(act())
    layers.append(nn.Linear(ic, oc, bias=bias))
    return layers

class ResBlock(nn.Module):
    def __init__(self, n_emb, ic, oc=None, ks=3, act=nn.SiLU, norm=nn.BatchNorm2d, attn_chans=0):
        super().__init__()
        self.emb_proj = nn.Linear(n_emb, oc*2)
        self.conv1 = unet_conv(ic, oc, ks, act=act, norm=norm)
        self.conv2 = unet_conv(oc, oc, ks, act=act, norm=norm)
        self.idconv = nn.Identity() if ic==oc else nn.Conv2d(ic, oc, kernel_size=1)
        self.attn = False
        if attn_chans: self.attn = SelfAttentionMultiHead(oc, attn_chans)
    
    def forward(self, x, t):
        inp = x
        x = self.conv1(x)
        emb = self.emb_proj(act(t))[:, :, None, None]
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x*(1+scale) + shift
        x = self.conv2(x) + self.idconv(inp)
        if self.attn: x = x + self.attn(x)
        return x

class UNET_Encoder(nn.Module):
    def __init__(self, n_emb, in_channels, nf, attn_chans):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        for f in nf:
            self.encoder.append(ResBlock(n_emb, in_channels, oc=f, attn_chans=attn_chans))
            in_channels = f
        
    def forward(self, x, t):
        skips = []
        for enc in self.encoder:
            x = enc(x, t)
            skips.append(x)
            x = self.pool(x)
        return x, skips

class UNET_Decoder(nn.Module):
    def __init__(self, n_emb, nf, attn_chans):
        super().__init__()
        self.decoder = nn.ModuleList()
        for f in nf:
            self.decoder.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.decoder.append(ResBlock(n_emb, f*2, oc=f, attn_chans=attn_chans))
    
    def forward(self, x, skips, t):
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            x = torch.cat((skips[i//2], x), dim=1)
            x = self.decoder[i+1](x, t)
        return x

class UNET(nn.Module):
    def __init__(self, n_classes, in_channels, out_channels, nf=[64, 128, 256, 512], attn_chans=8):
        super().__init__()
        self.t_emb = nf[0]
        n_emb = self.t_emb*4
        self.cond_emb = nn.Embedding(n_classes, n_emb)

        self.emb_mlp = nn.Sequential(
            lin(self.t_emb, n_emb, act=None, norm=nn.BatchNorm1d),
            lin(n_emb, n_emb)
        )

        self.unet_encoder = UNET_Encoder(n_emb, in_channels, nf, attn_chans)

        self.bottle_neck = ResBlock(n_emb, nf[-1], nf[-1])

        self.unet_decoder = UNET_Decoder(n_emb, nf[::-1], attn_chans)
        
        self.final_conv = nn.Conv2d(nf[0], out_channels, kernel_size=1)

    def forward(self, inp):
        x, t, c = inp
        temb = timestep_embedding(t, self.t_emb)
        cemb = self.cond_emb(c)
        emb = self.emb_mlp(temb) + cemb
        x, skips = self.unet_encoder(x, emb)
        x = self.bottle_neck(x, emb)
        x = self.unet_decoder(x, skips[::-1], emb)
        return self.final_conv(x)