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
    mask = mask.squeeze()

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

def get_heatmap2(mask, original_img=None, save_path=None, alpha=0.6):
    """
    Generate and optionally save a heatmap visualization of the mask.
    
    Parameters:
    -----------
    mask : numpy.ndarray
        The mask to visualize, can be single-channel or multi-channel
    original_img : numpy.ndarray, optional
        Original image to overlay the heatmap on
    save_path : str, optional
        Path to save the visualization
    show : bool, default=False
        Whether to display the plot
    alpha : float, default=0.6
        Transparency of the heatmap when overlaid on the original image
    """
    # Handle single-channel mask
    if mask.ndim == 2 or (mask.ndim == 3 and mask.shape[2] == 1):
        # If mask is already single-channel, just squeeze it if needed
        lum_img = mask.squeeze()
    else:
        # Original behavior for 3-channel masks
        lum_img = np.maximum(
            np.maximum(
                mask[:, :, 0],
                mask[:, :, 1],
            ),
            mask[:, :, 2],
        )
    
    # Get image dimensions
    if original_img is not None:
        height, width = original_img.shape[:2]
    else:
        height, width = lum_img.shape[:2]
    
    # Create figure with the exact pixel size of the image
    dpi = 100  # Default DPI
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Remove margins and padding
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    # If original image is provided, show it first
    if original_img is not None:
        if alpha < 1.0:
            plt.imshow(original_img)
            # Overlay heatmap with transparency
            imgplot = plt.imshow(lum_img, cmap="jet", alpha=alpha)
        else:
            # Create a blended visualization
            imgplot = plt.imshow(lum_img, cmap="jet")
    else:
        # Just show the heatmap
        imgplot = plt.imshow(lum_img, cmap="jet")
        
    # Add colorbar
    plt.colorbar(imgplot)
    
    # Save if path is provided
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    
    return

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
