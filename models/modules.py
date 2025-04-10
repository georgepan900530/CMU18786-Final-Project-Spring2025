import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


def get_heatmap(mask):
    lum_img = np.maximum(
        np.maximum(
            mask[:, :, 0],
            mask[:, :, 1],
        ),
        mask[:, :, 2],
    )
    imgplot = plt.imshow(lum_img)
    imgplot.set_cmap("jet")
    plt.colorbar()
    plt.axis("off")
    pylab.show()
    return


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

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(DSConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# Patchify the image into smaller patches for transformer.
# Reference from the implementation of ViT (https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py)


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, local_conv=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5
        self.local_conv = local_conv
        if local_conv:
            self.local_conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1)

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        if self.local_conv:
            # Optionally apply a convolution over the sequence dimension.
            # Rearrange from (b, n, dim) -> (b, dim, n)
            x_conv = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
            x = x + x_conv  # fuse local context

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, local_conv=False
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            local_conv=local_conv,
                        ),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class RainDropMaskDecoder(nn.Module):
    """
    RainDropMaskDecoder is a module that decodes the transformer latent features into the rain drop mask.

    Params
    ------
    embed_dim: int
        Dimension of the transformer latent features
    num_heads: int
        Number of heads in the transformer
    depth: int
        Number of layers in the transformer
    mlp_dim: int
        Dimension of the feed forward network
    dropout: float
        Dropout rate
    img_size: int or tuple
        Size of the input image (height, width)
    patch_size: int or tuple
        Size of the patches (height, width)
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        depth,
        mlp_dim,
        dropout=0.0,
        img_size=(224, 224),
        patch_size=16,
        channels=3,
        local_conv=False,
    ):
        super(RainDropMaskDecoder, self).__init__()
        self.params = {
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "depth": depth,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
            "local_conv": local_conv,
            "img_size": img_size,
            "patch_size": patch_size,
            "channels": channels,
        }        
        # Process image and patch sizes
        img_height, img_width = pair(img_size)
        patch_height, patch_width = pair(patch_size)

        # Calculate number of patches
        assert (img_height % patch_height == 0) and (
            img_width % patch_width == 0
        ), "Image size must be divisible by patch size"
        self.num_patches = (img_height // patch_height) * (img_width // patch_width)
        self.patch_dim = channels * patch_height * patch_width

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Transformer for processing encoded features
        self.transformer = Transformer(
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            dim_head=embed_dim // num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            local_conv=local_conv,
        )

        # Decoder layers to upsample back to original image size
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, patch_height * patch_width),
            Rearrange(
                "b (h w) (p1 p2) -> b 1 (h p1) (w p2)",
                h=img_height // patch_height,
                w=img_width // patch_width,
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Output a mask with values between 0 and 1
        )

    def forward(self, x):
        # x shape: [batch_size, num_patches, embed_dim]
        # print(x.shape)
        x = self.to_patch_embedding(x)
        x += self.pos_embedding
        x = self.transformer(x)
        # Decode the transformer output to image space
        mask = self.decoder(x)
        return mask


if __name__ == "__main__":
    rain_drop_mask_decoder = RainDropMaskDecoder(
        embed_dim=1024, num_heads=8, depth=12, mlp_dim=4096, dropout=0.0
    )
    input = torch.randn(1, 3, 224, 224)
    mask = rain_drop_mask_decoder(input)
    print(mask.shape)
    mask = mask[0].detach().cpu().numpy()
