import cv2
import numpy as np


def saturate_weaken_HSV(img, severity=1):
    """Saturate weaken by scaling S channel in HSV."""
    scale = [0.7, 0.55, 0.4, 0.2, 0.0][severity - 1]
    hsv = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), dtype=np.float32)
    hsv[:, :, 1] = scale * hsv[:, :, 1]
    hsv = np.uint8(np.clip(hsv, 0, 255))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def saturate_weaken_YCrCb(img, severity=1):
    """Saturate weaken by scaling CrCb channel in YCrCb."""
    scale = [0.6, 0.4, 0.2, 0.1, 0.0][severity - 1]
    ycrcb = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB), dtype=np.float32)
    ycrcb[:, :, 1] = 128 + (ycrcb[:, :, 1] - 128) * scale
    ycrcb[:, :, 2] = 128 + (ycrcb[:, :, 2] - 128) * scale
    ycrcb = np.uint8(np.clip(ycrcb, 0, 255))
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)
    return img


def saturate_strengthen_HSV(img, severity=1):
    """Saturate strengthen by scaling S channel in HSV."""
    scale = [3.0, 6.0, 12.0, 20.0, 64.0][severity - 1]
    hsv = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), dtype=np.float32)
    hsv[:, :, 1] = scale * hsv[:, :, 1]
    hsv = np.uint8(np.clip(hsv, 0, 255))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return img


def saturate_strengthen_YCrCb(img, severity=1):
    """Saturate strengthen by scaling CrCb channel in YCrCb."""
    scale = [2.0, 3.0, 5.0, 8.0, 16.0][severity - 1]
    ycrcb = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB), dtype=np.float32)
    ycrcb[:, :, 1] = 128 + (ycrcb[:, :, 1] - 128) * scale
    ycrcb[:, :, 2] = 128 + (ycrcb[:, :, 2] - 128) * scale
    ycrcb = np.uint8(np.clip(ycrcb, 0, 255))
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)
    return img
