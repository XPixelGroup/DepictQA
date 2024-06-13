import glob
import json
import os
import random
from shutil import copy

import numpy as np
from PIL import Image
from x_distortion import add_distortion, distortions_dict


def get_distortion_name(distortion_name):
    distortion_names = []
    for key in distortions_dict:
        distortion_names += distortions_dict[key]
    # distortion function name
    if distortion_name in distortion_names:
        pass
    # distortion category name
    elif distortion_name in distortions_dict:
        distortion_name = random.choice(distortions_dict[distortion_name])
    # others (random sample)
    else:
        key = random.choice(list(distortions_dict.keys()))
        distortion_name = random.choice(distortions_dict[key])
    return distortion_name


if __name__ == "__main__":
    resize = 224
    save_dir = "tests/res_multi_dists"
    os.makedirs(save_dir, exist_ok=True)
    img_path = "tests/test_image.png"
    distortions = list(distortions_dict.keys())

    for distortion1 in distortions:
        for distortion2 in distortions:
            if distortion1 == distortion2:
                continue
            img = Image.open(img_path).convert("RGB")
            h, w = img.height, img.width
            if resize < min(h, w):
                ratio = resize / min(h, w)
                h_new, w_new = round(h * ratio), round(w * ratio)
                img = img.resize((w_new, h_new), resample=Image.BICUBIC)
            img_lq = np.array(img)
            for distortion in [distortion1, distortion2]:
                if distortion == "oversharpen":
                    severity = 3
                else:
                    severity = 2
                distortion_name = get_distortion_name(distortion)
                img_lq = add_distortion(
                    img_lq, severity=severity, distortion_name=distortion_name
                )

            save_path = os.path.join(save_dir, f"{distortion1}_{distortion2}.png")
            img_lq = Image.fromarray(img_lq)
            img_lq.save(save_path)
