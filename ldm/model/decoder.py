import torch
from torch import nn
from attention import SelfAttention

class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        n, c, h, w = x.shape
        x = x.view(n, c, h*w)
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)
        x = x.view(n, c, h, w)
        x += residue
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.g_norm_1 = nn.GroupNorm(num_groups=32, num_channels=in_c)
        self.conv_1 = nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1)
        
        self.g_norm_2 = nn.GroupNorm(num_groups=in_c, num_channels=out_c)
        self.conv_2 = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3, padding=1)

        if in_c == out_c:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x
        x = self.g_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.g_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + self.residual_layer(residue)

class Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            nn.Conv2d(4, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            AttentionBlock(512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x /= 0.18215
        for module in self:
            x = module(x)
        return x