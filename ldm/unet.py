import torch
from torch import nn

from time_embedding import *
from attention import *

def unet_conv(in_channels, out_channels, kernel_size=3, stride=1, act=nn.SiLU, norm=None, bias=True):
    layers = nn.Sequential()
    if norm: layers.append(norm(in_channels))
    if act: layers.append(act())
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=bias))
    return layers

class ResBlock(nn.Module):
    def __init__(self, n_embedding, in_channels, out_channels=None, kernel_size=3, act=nn.SiLU, norm=nn.BatchNorm2d, attn_channs=0):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        self.emb_proj = nn.Linear(n_embedding, out_channels*2)
        self.conv_1 = unet_conv(in_channels, out_channels, kernel_size=kernel_size, act=act, norm=norm)
        self.conv_2 = unet_conv(out_channels, out_channels, kernel_size=kernel_size, act=act, norm=norm)
        self.id_conv = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.attn = False
        if attn_channs: self.attn = SelfAttention(out_channels, attn_channs)
        
    def forward(self, x, t):
        inp = x
        x = self.conv_1(x)
        emb = self.emb_proj(F.silu(t))[:, :, None, None]
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = x*(1+scale) + shift
        x = self.conv_2(x) 
        x += self.id_conv(inp)
        if self.attn:
            x += self.attn(x)
        return x

class DownSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1) 
    
    def forward(self, x):
        return self.conv(x)

class UNET_Encoder(nn.Module):
    def __init__(self, n_embedding, channels, attn_channs=0, attn_start=1):
        super().__init__()
        self.down_blocks = nn.ModuleList()
        self.down_sample = nn.ModuleList()
        
        n_resolutions = len(channels)
        out_channels = channels[0]
        for i in range(n_resolutions):
            in_channels = out_channels
            out_channels = channels[i]
            down = nn.ModuleList()
            for j in range(2):
                down.append(
                    ResBlock(
                        n_embedding, 
                        in_channels if j==0 else out_channels, 
                        out_channels=out_channels, 
                        attn_channs=0 if j<attn_start else attn_channs
                    )
                )
            self.down_blocks.append(down)
            self.down_sample.append(
                DownSample(out_channels) if (i < n_resolutions-1) and (j == 1) else nn.Identity()
            )
    
    def forward(self, x, t):
        skips = []
        for i in range(len(self.down_blocks)):
            for down in self.down_blocks[i]:
                skips.append(x)
                x = down(x, t)
            skips.append(x)
            x = self.down_sample[i](x)
        return x, skips

class UNET_Bottleneck(nn.Module):
    def __init__(self, n_embedding, in_channels):
        super().__init__()
        self.unet_bottleneck_1 = ResBlock(n_embedding, in_channels, attn_channs=8)
        self.unet_bottleneck_2 = ResBlock(n_embedding, in_channels)
    
    def forward(self, x, t):
        x = self.unet_bottleneck_1(x, t)
        return self.unet_bottleneck_2(x, t)

class UpSample(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2.0),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET_Decoder(nn.Module):
    def __init__(self, n_embedding, channels, attn_channs=0, attn_start=1):
        super().__init__()
        self.up_blocks = nn.ModuleList()
        self.up_sample = nn.ModuleList()

        n_resolutions = len(channels)
        out_channels = channels[0]
        for i in range(n_resolutions):
            prev_channels = out_channels
            in_channels = channels[min(i+1, n_resolutions-1)]
            out_channels = channels[i]
            up = nn.ModuleList()
            for j in range(3):
                up.append(
                    ResBlock(
                        n_embedding, 
                        (prev_channels if j==0 else out_channels) + (in_channels if j==2 else out_channels), 
                        out_channels=out_channels, 
                        attn_channs=0 if j>=n_resolutions-attn_start else attn_channs
                    )
                )
            self.up_blocks.append(up)
            self.up_sample.append(
                UpSample(out_channels) if (i < n_resolutions-1) and (j==2) else nn.Identity()
            )

    def forward(self, x, t, skips):
        for i in range(len(self.up_blocks)):
            for up in self.up_blocks[i]:
                x = up(torch.cat((x, skips.pop()), dim=1), t)
            x = self.up_sample[i](x)
        return x

class UNET(nn.Module):
    def __init__(self, n_classes, in_channels, out_channels, channels=(64, 128, 256, 512), attn_channs=8):
        super().__init__()
        self.n_channels = channels[0]
        self.n_embedding = self.n_channels * 4
        self.timestep_embedding = TimestepEmbedding(self.n_channels, self.n_embedding)
        self.condition_embedding = nn.Embedding(n_classes, self.n_embedding)

        self.conv_in = nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1)
        self.encoder = UNET_Encoder(self.n_embedding, channels, attn_channs=attn_channs)
        self.bottleneck = UNET_Bottleneck(self.n_embedding, channels[-1])
        self.decoder = UNET_Decoder(self.n_embedding, channels[::-1], attn_channs=attn_channs)
        self.conv_out = unet_conv(channels[0], out_channels, act=nn.SiLU, norm=nn.BatchNorm2d, bias=False)

    def forward(self, x, t, c):
        emb = self.timestep_embedding(t) + self.condition_embedding(c)
        x = self.conv_in(x)
        x, skips = self.encoder(x, emb)
        x = self.bottleneck(x, emb)
        x = self.decoder(x, emb, skips)
        x = self.conv_out(x)
        return x