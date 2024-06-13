import cv2
import numpy as np


def oversharpen(img, severity=1):
    """OverSharpening filter."""
    assert img.dtype == np.uint8, "Image array should have dtype of np.uint8"
    assert severity in [1, 2, 3, 4, 5], "Severity must be an integer between 1 and 5."

    amount = [2, 2.8, 4, 6, 8][severity - 1]
    # Create a blurred/smoothed version
    blur_radius = 5
    sigmaX = 0
    blurred = cv2.GaussianBlur(img, (blur_radius, blur_radius), sigmaX)
    # (1 + amount) * img - amount * blurred -> enhance high frequency, keep low frequency
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened
