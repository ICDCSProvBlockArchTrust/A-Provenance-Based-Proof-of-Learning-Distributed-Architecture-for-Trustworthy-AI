import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):    # x: (B, C, H, W)
        return self.proj(x)  # x: (B, dim, H//patch, W//patch)

class ViTBlock(nn.Module):
    def __init__(self, dim, attn_cls):
        super().__init__()
        self.attn = attn_cls
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.ff = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )

    def forward(self, x):
        x = self.attn(x) + x
        x = self.norm1(x)
        x = self.ff(x) + x
        x = self.norm2(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, dim, depth, attn_fn, patch_size=2, img_size=32, channels=3):
        super().__init__()
        self.patch_embed = PatchEmbedding(channels, dim, patch_size)
        H, W = img_size // patch_size, img_size // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, dim, H, W)) 

        self.blocks = nn.Sequential(*[ViTBlock(dim, attn_fn) for _ in range(depth)])

        self.output_proj = nn.Sequential(
            nn.ConvTranspose2d(dim, channels, kernel_size=patch_size, stride=patch_size)
        )

    def forward(self, x):
        x = self.patch_embed(x)       # (B, dim, H', W')
        x = x + self.pos_embed        # (B, dim, H', W')
        x = self.blocks(x)            # process with 4D-attn blocks
        x = self.output_proj(x)       # (B, in_channels, H, W) — reconstructed shape
        return x