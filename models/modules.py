import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#  Depthwise Separable Convolution
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
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = embed_dim // num_heads
