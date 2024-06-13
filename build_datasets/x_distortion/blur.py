import cv2
import numpy as np
from skimage.filters import gaussian

from .helper import (
    clipped_zoom,
    gen_disk,
    gen_lensmask,
    motion_blur,
    shuffle_pixels_njit,
)


def blur_gaussian(img, severity=1):
    """Gaussian blur."""
    sigma = [1, 2, 3, 4, 5][severity - 1]
    img = np.array(img) / 255.0
    img = gaussian(img, sigma=sigma, channel_axis=-1)
    img = np.clip(img, 0, 1) * 255
    return img


def blur_gaussian_lensmask(img, severity=1):
    """Gaussian blur with lens mask."""
    gamma, sigma = [(2.0, 2), (2.4, 4), (3.0, 6), (3.8, 8), (5.0, 10)][severity - 1]
    img_orig = np.array(img) / 255.0
    h, w = img.shape[:2]
    mask = gen_lensmask(h, w, gamma=gamma)[:, :, None]
    img = gaussian(img_orig, sigma=sigma, channel_axis=-1)
    img = mask * img_orig + (1 - mask) * img
    img = np.clip(img, 0, 1) * 255
    return img


def blur_motion(img, severity=1):
    """Motion blur."""
    radius, sigma = [(5, 3), (10, 5), (15, 7), (15, 9), (20, 12)][severity - 1]
    angle = np.random.uniform(-90, 90)
    img = np.array(img)
    img = motion_blur(img, radius=radius, sigma=sigma, angle=angle)
    img = np.clip(img, 0, 255)
    return img


def blur_glass(img, severity=1):
    """Glass blur."""
    sigma, shift, iteration = [
        (0.7, 1, 1),
        (0.9, 2, 1),
        (1.2, 2, 2),
        (1.4, 3, 2),
        (1.6, 4, 2),
    ][severity - 1]
    img = np.array(img) / 255.0
    img = gaussian(img, sigma=sigma, channel_axis=-1)
    img = shuffle_pixels_njit(img, shift=shift, iteration=iteration)
    img = np.clip(gaussian(img, sigma=sigma, channel_axis=-1), 0, 1) * 255
    return img


def blur_lens(img, severity=1):
    """Lens blur."""
    radius = [2, 3, 4, 6, 8][severity - 1]
    img = np.array(img) / 255.0
    kernel = gen_disk(radius=radius)
    img_lq = []
    for i in range(3):
        img_lq.append(cv2.filter2D(img[:, :, i], -1, kernel))
    img_lq = np.array(img_lq).transpose((1, 2, 0))
    img_lq = np.clip(img_lq, 0, 1) * 255
    return img_lq


def blur_zoom(img, severity=1):
    """Zoom blur."""
    zoom_factors = [
        np.arange(1, 1.03, 0.02),
        np.arange(1, 1.06, 0.02),
        np.arange(1, 1.10, 0.02),
        np.arange(1, 1.15, 0.02),
        np.arange(1, 1.21, 0.02),
    ][severity - 1]
    img = (np.array(img) / 255.0).astype(np.float32)
    h, w = img.shape[:2]
    img_lq = np.zeros_like(img)
    for zoom_factor in zoom_factors:
        zoom_layer = clipped_zoom(img, zoom_factor)
        img_lq += zoom_layer[:h, :w, :]
    img_lq = (img + img_lq) / (len(zoom_factors) + 1)
    img_lq = np.clip(img_lq, 0, 1) * 255
    return img_lq


def blur_jitter(img, severity=1):
    """Jitter blur."""
    shift = [1, 2, 3, 4, 5][severity - 1]
    img = np.array(img)
    img_lq = shuffle_pixels_njit(img, shift=shift, iteration=1)
    return np.uint8(img_lq)
