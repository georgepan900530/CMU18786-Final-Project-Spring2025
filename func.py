"""
This file is refactored from the unofficial implementation of the paper (https://github.com/shleecs/DeRaindrop_unofficial)
"""

import cv2
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from torch.autograd import Variable


def plot_raindrop_mask(mask, save_path=None, show=False):
    """
    Plot a heatmap visualization of the raindrop mask.

    Parameters:
    -----------
    mask : torch.Tensor or numpy.ndarray
        The raindrop mask with shape (1, H, W) or (H, W)
    save_path : str, optional
        Path to save the visualization. If None, the image is not saved.
    show : bool, default=True
        Whether to display the plot
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    # Squeeze the channel dimension if it exists
    if mask.shape[0] == 1:
        mask = mask.squeeze(0)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Plot heatmap
    im = plt.imshow(mask, cmap="jet")
    plt.colorbar(im, label="Mask Intensity")
    plt.title("Raindrop Mask Heatmap")
    plt.axis("off")

    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    # Show or close
    if show:
        plt.show()
    else:
        plt.close()


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


def get_mask(dg_img, img):
    # downgraded image - image
    mask = np.fabs(dg_img - img)
    # threshold under 30
    mask[np.where(mask < (30.0 / 255.0))] = 0.0
    mask[np.where(mask > 0.0)] = 1.0
    # avg? max?
    # mask = np.average(mask, axis=2)
    mask = np.max(mask, axis=2)
    mask = np.expand_dims(mask, axis=2)
    return mask


def torch_variable(x, is_train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if is_train:
        return Variable(
            torch.from_numpy(np.array(x).transpose((0, 3, 1, 2))), requires_grad=True
        ).to(device)
    else:
        # Replace volatile=True with torch.no_grad()
        with torch.no_grad():
            return Variable(torch.from_numpy(np.array(x).transpose((0, 3, 1, 2)))).to(
                device
            )
