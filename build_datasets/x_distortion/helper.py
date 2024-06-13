import math

import numpy as np
from numba import njit
from scipy.ndimage import zoom as scizoom


def gen_lensmask(h, w, gamma):
    """For blur_gaussian_lensmask & brightness_vignette."""
    dist1 = np.array([list(range(w))] * h) - w // 2
    dist2 = np.array([list(range(h))] * w) - h // 2
    dist2 = np.transpose(dist2, (1, 0))
    dist = np.sqrt((dist1**2 + dist2**2)) / np.sqrt((w**2 + h**2) / 4)
    mask = (1 - dist) ** gamma
    return mask


def gen_disk(radius, dtype=np.float32):
    """For blur_lens."""
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
    else:
        L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    disk /= np.sum(disk)
    return disk


def clipped_zoom(img, zoom_factor):
    """For blur_zoom."""
    # clipping along the width dimension
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2
    # clipping along the height dimension
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2
    img = scizoom(
        img[top0 : top0 + ch0, top1 : top1 + ch1],
        (zoom_factor, zoom_factor, 1),
        order=1,
    )
    return img


def get_motion_blur_kernel(width, sigma):
    def gauss_function(x, mean, sigma):
        return (np.exp(-((x - mean) ** 2) / (2 * (sigma**2)))) / (
            np.sqrt(2 * np.pi) * sigma
        )

    kernel = gauss_function(np.arange(width), 0, sigma)
    kernel = kernel / np.sum(kernel)
    return kernel


def shift_img(img, dx, dy):
    if dx < 0:
        shifted = np.roll(img, shift=img.shape[1] + dx, axis=1)
        shifted[:, dx:] = shifted[:, dx - 1 : dx]
    elif dx > 0:
        shifted = np.roll(img, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx : dx + 1]
    else:
        shifted = img

    if dy < 0:
        shifted = np.roll(shifted, shift=img.shape[0] + dy, axis=0)
        shifted[dy:, :] = shifted[dy - 1 : dy, :]
    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy : dy + 1, :]
    return shifted


def motion_blur(x, radius, sigma, angle):
    """For blur_motion."""
    width = radius * 2 + 1
    kernel = get_motion_blur_kernel(width, sigma)
    point = (width * np.sin(np.deg2rad(angle)), width * np.cos(np.deg2rad(angle)))
    hypot = math.hypot(point[0], point[1])

    blurred = np.zeros_like(x, dtype=np.float32)
    for i in range(width):
        dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
        dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
        if np.abs(dy) >= x.shape[0] or np.abs(dx) >= x.shape[1]:
            # simulated motion exceeded img borders
            break
        shifted = shift_img(x, dx, dy)
        blurred = blurred + kernel[i] * shifted
    return blurred


# Numba nopython compilation to shuffle_pixles
@njit()
def shuffle_pixels_njit(img, shift, iteration):
    """For blur_glass & blur_jitter."""
    height, width = img.shape[:2]
    # locally shuffle pixels
    for _ in range(iteration):
        for h in range(height - shift, shift, -1):
            for w in range(width - shift, shift, -1):
                dx, dy = np.random.randint(-shift, shift, size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                img[h, w], img[h_prime, w_prime] = img[h_prime, w_prime], img[h, w]
    return img
