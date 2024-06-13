import numpy as np
from PIL import Image, ImageEnhance


def contrast_weaken_scale(img, severity=1):
    """Contrast weaken by scaling."""
    factor = [0.75, 0.6, 0.45, 0.3, 0.2][severity - 1]
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    img = np.uint8(np.clip(np.array(img), 0, 255))
    return img


def contrast_weaken_stretch(img, severity=1):
    """Contrast weaken by stretching."""
    factor = [1.0, 0.9, 0.8, 0.6, 0.4][severity - 1]
    img = np.array(img) / 255.0
    img_mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = 1.0 / (1 + (img_mean / (img + 1e-12)) ** factor)
    img = np.uint8(np.clip(img, 0, 1) * 255)
    return img


def contrast_strengthen_scale(img, severity=1):
    """Contrast strengthen by scaling."""
    factor = [1.4, 1.7, 2.1, 2.6, 4.0][severity - 1]
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(factor)
    img = np.uint8(np.clip(np.array(img), 0, 255))
    return img


def contrast_strengthen_stretch(img, severity=1):
    """Contrast strengthen by stretching."""
    factor = [2.0, 4.0, 6.0, 8.0, 10.0][severity - 1]
    img = np.array(img) / 255.0
    img_mean = np.mean(img, axis=(0, 1), keepdims=True)
    img = 1.0 / (1 + (img_mean / (img + 1e-12)) ** factor)
    img = np.uint8(np.clip(img, 0, 1) * 255)
    return img
