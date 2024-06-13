import cv2
import numpy as np

from .helper import gen_lensmask


def brightness_brighten_shfit_HSV(img, severity=1):
    """Mean shift V channel in HSV."""
    shift = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    img = np.float32(np.array(img) / 255.0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] += shift
    img_lq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def brightness_brighten_shfit_RGB(img, severity=1):
    """Mean shift RGB."""
    shift = [0.1, 0.15, 0.2, 0.27, 0.35][severity - 1]
    img = np.float32(np.array(img) / 255.0)
    img_lq = img + shift
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def brightness_brighten_gamma_HSV(img, severity=1):
    """Enhance V channel in HSV with a gamma function."""
    gamma = [0.7, 0.58, 0.47, 0.36, 0.25][severity - 1]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv = np.array(img_hsv / 255.0)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] ** gamma
    img_lq = np.uint8(np.clip(img_hsv, 0, 1) * 255.0)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_HSV2RGB)
    return img_lq


def brightness_brighten_gamma_RGB(img, severity=1):
    """Enhance RGB with a gamma function."""
    gamma = [0.8, 0.7, 0.6, 0.45, 0.3][severity - 1]
    img = np.array(img / 255.0)
    img_lq = img**gamma
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def brightness_darken_shfit_HSV(img, severity=1):
    """Mean shift V channel in HSV."""
    shift = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    img = np.float32(np.array(img) / 255.0)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] -= shift
    img_lq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def brightness_darken_shfit_RGB(img, severity=1):
    """Mean shift RGB."""
    shift = [0.1, 0.15, 0.2, 0.27, 0.35][severity - 1]
    img = np.float32(np.array(img) / 255.0)
    img_lq = img - shift
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def brightness_darken_gamma_HSV(img, severity=1):
    """Reduce V channel in HSV with a gamma function."""
    gamma = [1.5, 1.8, 2.2, 2.7, 3.5][severity - 1]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv = np.array(img_hsv / 255.0)
    img_hsv[:, :, 2] = img_hsv[:, :, 2] ** gamma
    img_lq = np.uint8(np.clip(img_hsv, 0, 1) * 255.0)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_HSV2RGB)
    return img_lq


def brightness_darken_gamma_RGB(img, severity=1):
    """Reduce RGB with a gamma function."""
    gamma = [1.4, 1.7, 2.1, 2.6, 3.2][severity - 1]
    img = np.array(img / 255.0)
    img_lq = img**gamma
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def brightness_vignette(img, severity=1):
    """Vignette effect in RGB."""
    gamma = [0.5, 0.875, 1.25, 1.625, 2][severity - 1]
    img = np.array(img)
    h, w = img.shape[:2]
    mask = gen_lensmask(h, w, gamma=gamma)[:, :, None]
    img_lq = mask * img
    return np.uint8(np.clip(img_lq, 0, 255))
