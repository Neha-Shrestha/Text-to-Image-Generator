class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Features or channels, H, W)
        residue = x
        n, c, h, w = x.shape
        # (B, Features, H, W) -> (B, Features, H * W)
        x = x.view(n, c, h*w)
        # (B, Features, H * W) -> (B, H * W, Features)
        x = x.transpose(-1, -2)
        # (B, Features, H * W) -> (B, H * W, Features)
        x = self.attention(x)
        # (B, H * W, Features) -> (B, Features, H * W)
        x = x.transpose(-1, -2)
        # (B, Features, H * W) -> (B, Features, H, W)
        x = x.view(n, c, h, w)
        
        x += residue
        return x
