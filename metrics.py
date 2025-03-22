import skimage
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import lpips
import torch
import numpy as np

loss_fn = lpips.LPIPS(net="alex", spatial=True)
loss_fn.cuda()


def calc_psnr(im1, im2):
    return peak_signal_noise_ratio(im1, im2)


def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return ssim(im1_y, im2_y)


def im2tensor(image, imtype=np.uint8, cent=1.0, factor=255.0 / 2.0):
    return torch.Tensor(
        (image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
    )


def calc_lpips(img1, img2):
    ref = img1[:, :, ::-1]
    output = img2
    output = cv2.resize(
        img2, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA
    )
    output = output[:, :, ::-1]
    ref = im2tensor(ref)
    output = im2tensor(output)
    ref = ref.cuda()
    output = output.cuda()
    d = loss_fn.forward(ref, output)
    return d.mean().cpu().detach().numpy()
