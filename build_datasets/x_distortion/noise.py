import cv2
import numpy as np
import skimage as sk


def noise_gaussian_RGB(img, severity=1):
    """Additive Gaussian noise."""
    sigma = [0.05, 0.1, 0.15, 0.2, 0.25][severity - 1]
    img = np.array(img) / 255.0
    noise = np.random.normal(0, sigma, img.shape)
    img_lq = img + noise
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def noise_gaussian_YCrCb(img, severity=1):
    """Additive Gaussian noise with higher noise in color channels."""
    sigma_y = [0.05, 0.06, 0.07, 0.08, 0.09][severity - 1]
    sigma_r = sigma_y * [1, 1.45, 1.9, 2.35, 2.8][severity - 1]
    sigma_b = sigma_y * [1, 1.45, 1.9, 2.35, 2.8][severity - 1]
    h, w = img.shape[:2]
    img = np.float32(np.array(img) / 255.0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    noise_l = np.expand_dims(np.random.normal(0, sigma_y, (h, w)), 2)
    noise_r = np.expand_dims(np.random.normal(0, sigma_r, (h, w)), 2)
    noise_b = np.expand_dims(np.random.normal(0, sigma_b, (h, w)), 2)
    noise = np.concatenate((noise_l, noise_r, noise_b), axis=2)
    img_lq = np.float32(img + noise)
    img_lq = cv2.cvtColor(img_lq, cv2.COLOR_YCR_CB2RGB)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def noise_speckle(img, severity=1):
    """Multiplicative Gaussian noise."""
    scale = [0.14, 0.21, 0.28, 0.35, 0.42][severity - 1]
    img = np.array(img) / 255.0
    noise = img * np.random.normal(size=img.shape, scale=scale)
    img_lq = img + noise
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def noise_spatially_correlated(img, severity=1):
    """Spatially correlated noise."""
    sigma = [0.08, 0.11, 0.14, 0.18, 0.22][severity - 1]
    img = np.array(img) / 255.0
    noise = np.random.normal(0, sigma, img.shape)
    img_lq = img + noise
    img_lq = cv2.blur(img_lq, [3, 3])
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def noise_poisson(img, severity=1):
    """Poisson noise."""
    factor = [80, 60, 40, 25, 15][severity - 1]
    img = np.array(img) / 255.0
    img_lq = np.random.poisson(img * factor) / float(factor)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)


def noise_impulse(img, severity=1):
    """Impulse noise / salt & pepper noise."""
    amount = [0.01, 0.03, 0.05, 0.07, 0.10][severity - 1]
    img = np.array(img) / 255.0
    img_lq = sk.util.random_noise(img, mode='s&p', amount=amount)
    return np.uint8(np.clip(img_lq, 0, 1) * 255.0)
