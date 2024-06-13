import numpy as np
from PIL import Image


def pixelate(img, severity=1):
    """Pixelate."""
    scale = [0.5, 0.4, 0.3, 0.25, 0.2][severity - 1]
    h, w = np.array(img).shape[:2]
    img = Image.fromarray(img)
    img = img.resize((int(w * scale), int(h * scale)), Image.BOX)
    img = img.resize((w, h), Image.NEAREST)
    img = np.uint8(np.clip(np.array(img), 0, 255))
    return img
