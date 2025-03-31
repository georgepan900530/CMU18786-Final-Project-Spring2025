import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange


class DSConv(nn.Module):
    """
    Depthwise Separable Convolution.
    Note that depthwise separable convolution is made up of depthwise and pointwise convolutions.
    The depthwise convolution is applied to each channel separately (setting groups=in_channels),
    and the pointwise convolution is applied to the entire feature map with kernel size equal to 1.

    Params
    ------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    kernel_size: int
        Kernel size of the depthwise convolution
    stride: int
        Stride of the depthwise convolution
    padding: int
        Padding of the depthwise convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Patchify the image into smaller patches for transformer.
# Reference from the implementation of ViT (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)
class Patchify(nn.Module):
    """
    Patchify the image into smaller patches for transformer, and then embed the patches into a vector.

    Params
    ------
    patch_height: int
        Height of the patch
    patch_width: int
        Width of the patch
    embed_dim: int
        Dimension of the embedded patches
    num_channels: int
        Number of channels in the image

    Returns
    -------
    patches: Tensor
        Tensor of shape (B, num_patches, embed_dim)
    """

    def __init__(self, patch_height, patch_width, embed_dim, num_channels=3):
        super(Patchify, self).__init__()
        self.patch_height = patch_height
        self.patch_width = patch_width
        patch_dim = num_channels * patch_height * patch_width
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",  # Rearrange the image into patches
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, img):
        return self.to_patch_embedding(img)


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head attention block.

    Params
    ------
    embed_dim: int
        Dimension of the embedded patches
    num_heads: int
        Number of attention heads

    Returns
    -------
    out: Tensor
        Tensor of shape (B, N, embed_dim)
    """

    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, q, k, v):
        B, N, D = q.shape
        B, T, D = k.shape
        assert k.shape == v.shape
        H = self.num_heads
        assert D % H == 0, "embed_dim must be divisible by num_heads"
        d = D // H

        q = self.head_proj(q).view(B, N, H, d).transpose(1, 2)
        k = self.head_proj(k).view(B, T, H, d).transpose(1, 2)
        v = self.head_proj(v).view(B, T, H, d).transpose(1, 2)

        dot_product = q @ k.transpose(-2, -1)
        dot_product = dot_product / np.sqrt(d)
        dot_product = F.softmax(dot_product, dim=-1)
        out = dot_product @ v
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        MultiHeadAttentionBlock(dim, num_heads=heads),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
