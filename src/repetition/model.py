
import torch
import torch.nn as nn
import torch.nn.functional as F

# Taken from 
# https://github.com/paulagd/SE_attention/blob/master/model.py
class SqueezeExcitationModule(nn.Module):
    def __init__(self, C, r=2):
        super().__init__()
        self.fc1 = nn.Linear(C, C // r, bias=False)
        self.fc2 = nn.Linear(C // r, C, bias=False)

    def forward(self, x):
        # x input shape: [Batch, Tokens, Channels] (B, N, C)
        # Global Average Pooling
        z = x.mean(dim=1) # Shape: [B, C]
        # 2. Excitation
        z = self.fc1(z)
        z = F.relu(z, inplace=True)
        z = self.fc2(z)
        z = torch.sigmoid(z) # Shape: [B, C]
        return x * z.unsqueeze(1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.se = SqueezeExcitationModule(dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        attn_out = self.se(attn_out)
        x = residual + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=128,
        depth=6,
        heads=4,
        mlp_dim=256
    ):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        self.blocks = nn.Sequential(
            *[TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.patch_size = patch_size

    def forward(self, x):
        B, C, _, _ = x.shape
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, C, -1, p, p)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.flatten(2) # [B, num_patches, patch_dim]

        x = self.patch_embed(x) # [B, num_patches, dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])