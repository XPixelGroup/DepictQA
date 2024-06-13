import argparse
import glob
import json
import os
import random
from shutil import copy

import numpy as np
from PIL import Image
from x_distortion import add_distortion, distortions_dict

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument(
    "-d",
    "--distortion_name",
    type=str,
    default=None,
    help="(1) distortion function name, (2) distortion category name, (3) others (random sample).",
)
parser.add_argument(
    "-a",
    "--all_severity",
    action="store_true",
    help="If False (default), random select one severity; else, all severities are used.",
)
parser.add_argument(
    "-p",
    "--prob_distortion",
    type=float,
    default=0.95,
    help="Probability of adding distortion.",
)
parser.add_argument(
    "-ns",
    "--noskip",
    action="store_true",
    help="If False (default), skip the same reference; else, regenerate with another name.",
)
parser.add_argument("--reference_dir", type=str, required=True)
parser.add_argument("--distortion_dir", type=str, required=True)
parser.add_argument("--meta_dir", type=str, required=True)
parser.add_argument("--seed", type=int, default=131)


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


def get_distortion_class(distortion_name):
    for key in distortions_dict:
        if distortion_name in distortions_dict[key]:
            return key


def seed_everything(seed=131):
    np.random.seed(seed)
    random.seed(seed**2)


if __name__ == "__main__":
    args = parser.parse_args()
    idx_start = 0
    idx_end = 5
    num_severity = 5
    resize = 224
    seed_everything(seed=args.seed)
    reference_dir = args.reference_dir
    img_paths = sorted(glob.glob(os.path.join(reference_dir, "*.png")))
    distortion_dir = args.distortion_dir
    meta_dir = args.meta_dir
    os.makedirs(distortion_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    for idx_ref, img_path in enumerate(img_paths[idx_start:idx_end]):
        print("=" * 100)
        print(idx_start + idx_ref)

        # check the first distortion img & json
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_lq_exist = os.path.exists(os.path.join(distortion_dir, f"{img_name}_0.png"))
        json_lq_exist = os.path.exists(os.path.join(meta_dir, f"{img_name}_0.json"))
        if img_lq_exist and json_lq_exist and not args.noskip:
            print(f"{img_path} has been generated, skip.")
            continue

        distortion_name = get_distortion_name(args.distortion_name)
        distortion_class = get_distortion_class(distortion_name)

        if random.random() > args.prob_distortion:
            severities = [0]
        elif args.all_severity:
            severities = list(range(1, num_severity + 1))
        else:
            severities = [random.randint(1, num_severity)]

        for severity in severities:
            idx = 0
            while True:
                save_path = os.path.join(distortion_dir, f"{img_name}_{idx}.png")
                save_json = os.path.join(meta_dir, f"{img_name}_{idx}.json")
                if not (os.path.exists(save_path) and os.path.exists(save_json)):
                    break
                idx += 1

            if severity == 0:
                copy(img_path, save_path)
                distortion_name = distortion_class = None
            else:
                img = Image.open(img_path).convert("RGB")
                h, w = img.height, img.width
                if resize < min(h, w):
                    ratio = resize / min(h, w)
                    h_new, w_new = round(h * ratio), round(w * ratio)
                    img = img.resize((w_new, h_new), resample=Image.BICUBIC)
                img = np.array(img)
                img_lq = add_distortion(
                    img, severity=severity, distortion_name=distortion_name
                )
                img_lq = Image.fromarray(img_lq)
                img_lq.save(save_path)

            meta = {
                "img_ref": img_path,
                "img_lq": save_path,
                "distortion_class": distortion_class,
                "distortion_name": distortion_name,
                "severity": severity,
            }
            with open(save_json, "w") as fw:
                fw.write(json.dumps(meta, indent=4))

            print(meta)
