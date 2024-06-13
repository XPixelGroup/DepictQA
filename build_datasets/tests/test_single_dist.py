import argparse
import os

import numpy as np
from PIL import Image
from x_distortion import add_distortion

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("-d", "--distortion_name")


if __name__ == "__main__":
    args = parser.parse_args()
    distortion_name = args.distortion_name

    resize = 224
    save_dir = "tests/res_single_dist"
    os.makedirs(save_dir, exist_ok=True)
    num_severity = 5
    img_path = "tests/test_image.png"
    img = Image.open(img_path).convert("RGB")
    h, w = img.height, img.width
    ratio = resize / min(h, w)
    h_new, w_new = round(h * ratio), round(w * ratio)
    img = img.resize((w_new, h_new), resample=Image.BICUBIC)
    img = np.array(img)

    for severity in range(num_severity):
        img_lq = add_distortion(
            img, severity=severity + 1, distortion_name=distortion_name
        )
        img_lq = Image.fromarray(img_lq)
        save_path = os.path.join(save_dir, f"{distortion_name}_{severity + 1}.png")
        img_lq.save(save_path)
