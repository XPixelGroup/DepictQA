import argparse
import glob
import json
import os
import random
from shutil import copy

import numpy as np
from PIL import Image
from scripts.constants_md import multi_distortions_dict
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
parser.add_argument("--exclude_json", type=str, default=None)
parser.add_argument("--split_json", type=str, required=True)
parser.add_argument("--reference_dir", type=str, required=True)
parser.add_argument("--distortion_dir", type=str, required=True)
parser.add_argument("--meta_dir", type=str, required=True)
parser.add_argument("--training", action="store_true")
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
    num_distortion = 2
    num_severity = 3
    resize = 224
    training = args.training
    seed_everything(seed=args.seed)

    split_json = args.split_json
    with open(split_json) as fr:
        split_dict = json.load(fr)
    img_names = split_dict["train"] if training else split_dict["test"]

    img_names_exclude = []
    exclude_json = args.exclude_json
    if exclude_json:
        with open(exclude_json) as fr:
            img_names_exclude = json.load(fr)

    print(f"All: {len(img_names)}")
    print(f"Exclude: {len(img_names_exclude)}")
    img_names = sorted(list(set(img_names) - set(img_names_exclude)))
    print(f"After Exclude: {len(img_names)}")
    random.shuffle(img_names)

    reference_dir = args.reference_dir
    img_paths = [os.path.join(reference_dir, img_name) for img_name in img_names]
    distortion_dir = args.distortion_dir
    meta_dir = args.meta_dir
    os.makedirs(distortion_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    for idx_ref, img_path in enumerate(img_paths[idx_start:idx_end]):
        print("=" * 100)
        print(idx_start + idx_ref)
        catch_error = False

        # check the first distortion img & json
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_lq_exist = os.path.exists(os.path.join(distortion_dir, f"{img_name}_0.png"))
        json_lq_exist = os.path.exists(os.path.join(meta_dir, f"{img_name}_0.json"))
        if img_lq_exist and json_lq_exist and not args.noskip:
            print(f"{img_path} has been generated, skip.")
            continue

        idx = 0
        while True:
            save_path = os.path.join(distortion_dir, f"{img_name}_{idx}.png")
            save_json = os.path.join(meta_dir, f"{img_name}_{idx}.json")
            if not (os.path.exists(save_path) and os.path.exists(save_json)):
                break
            idx += 1

        if random.random() > args.prob_distortion:
            copy(img_path, save_path)
            severities = 0
            distortion_names = distortion_classes = None
        else:
            img = Image.open(img_path).convert("RGB")
            h, w = img.height, img.width
            if resize < min(h, w):
                ratio = resize / min(h, w)
                h_new, w_new = round(h * ratio), round(w * ratio)
                img = img.resize((w_new, h_new), resample=Image.BICUBIC)
            img_lq = np.array(img)

            while True:
                distortion_classes = random.sample(
                    list(distortions_dict.keys()), num_distortion
                )
                dist1 = distortion_classes[0]
                dist2 = distortion_classes[1]
                if dist2 in multi_distortions_dict[dist1]:
                    break

            try:
                distortion_names = []
                severities = []
                print(distortion_classes)
                for distortion_class in distortion_classes:
                    distortion_name = get_distortion_name(distortion_class)
                    severity = random.randint(1, num_severity)
                    distortion_names.append(distortion_name)
                    severities.append(severity)
                    img_lq = add_distortion(
                        img_lq, severity=severity, distortion_name=distortion_name
                    )

                img_lq = Image.fromarray(img_lq)
                img_lq.save(save_path)

            except:
                assert "quantization_otsu" in distortion_names
                catch_error = True

        if not catch_error:
            meta = {
                "img_ref": img_path,
                "img_lq": save_path,
                "distortion_classes": distortion_classes,
                "distortion_names": distortion_names,
                "severities": severities,
            }
            with open(save_json, "w") as fw:
                fw.write(json.dumps(meta, indent=4))

            print(meta)
