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


def get_distortion(prob_distortion):
    if random.random() > prob_distortion:
        severities = 0
        distortion_names = distortion_classes = None
        return distortion_names, distortion_classes, severities

    num_distortion = random.choice([1, 2])  # num_distortion should be 1 or 2
    if num_distortion == 1:
        num_severity = 5
        distortion_name, distortion_class = get_distortion_name(None)
        severity = random.randint(1, num_severity)
        distortion_names = [distortion_name]
        distortion_classes = [distortion_class]
        severities = [severity]
    else:
        num_severity = 3
        while True:
            distortion_classes = random.sample(list(distortions_dict.keys()), 2)
            dist1 = distortion_classes[0]
            dist2 = distortion_classes[1]
            if dist2 in multi_distortions_dict[dist1]:
                break
        distortion_names = []
        severities = []
        for distortion_class in distortion_classes:
            distortion_name, _ = get_distortion_name(distortion_class)
            severity = random.randint(1, num_severity)
            distortion_names.append(distortion_name)
            severities.append(severity)
    return distortion_names, distortion_classes, severities


def get_distortion_name(distortion_name):
    # sample distortion_name
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
    # sample distortion_class
    for key in distortions_dict:
        if distortion_name in distortions_dict[key]:
            distortion_class = key
            break
    return distortion_name, distortion_class


def gen_distortion_img(img_path, save_path, distortion_names, severities, resize):
    if severities == 0:
        copy(img_path, save_path)
    else:
        img = Image.open(img_path).convert("RGB")
        h, w = img.height, img.width
        if resize < min(h, w):
            ratio = resize / min(h, w)
            h_new, w_new = round(h * ratio), round(w * ratio)
            img = img.resize((w_new, h_new), resample=Image.BICUBIC)
        img_lq = np.array(img)
        for distortion_name, severity in zip(distortion_names, severities):
            img_lq = add_distortion(
                img_lq, severity=severity, distortion_name=distortion_name
            )
        img_lq = Image.fromarray(img_lq)
        img_lq.save(save_path)
    return


def seed_everything(seed=131):
    np.random.seed(seed)
    random.seed(seed**2)


if __name__ == "__main__":
    args = parser.parse_args()
    idx_start = 0
    idx_end = 5
    resize = 224
    seed_everything(seed=args.seed)

    training = args.training
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

        # check the first distortion img & json
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        imgA_lq_exist = os.path.exists(
            os.path.join(distortion_dir, f"{img_name}_A0.png")
        )
        imgB_lq_exist = os.path.exists(
            os.path.join(distortion_dir, f"{img_name}_B0.png")
        )
        json_lq_exist = os.path.exists(os.path.join(meta_dir, f"{img_name}_0.json"))

        if imgA_lq_exist and imgB_lq_exist and json_lq_exist and not args.noskip:
            print(f"{img_path} has been generated, skip.")
            continue

        while True:
            distA_names, distA_classes, severities_A = get_distortion(
                args.prob_distortion
            )
            distB_names, distB_classes, severities_B = get_distortion(
                args.prob_distortion
            )
            severities_A_tmp = severities_A if severities_A else [0]
            severities_B_tmp = severities_B if severities_B else [0]
            # 1. At least one sample with multiple distortions
            # 2. if distortion_class is the same and severity is the same, resample
            if (len(severities_A_tmp) >= 2 or len(severities_B_tmp) >= 2) and (
                distA_classes != distB_classes or severities_A != severities_B
            ):
                break

        idx = 0
        while True:
            imgA_path = os.path.join(distortion_dir, f"{img_name}_A{idx}.png")
            imgB_path = os.path.join(distortion_dir, f"{img_name}_B{idx}.png")
            save_json = os.path.join(meta_dir, f"{img_name}_{idx}.json")
            if not (
                os.path.exists(imgA_path)
                and os.path.join(imgB_path)
                and os.path.exists(save_json)
            ):
                break
            idx += 1

        gen_distortion_img(img_path, imgA_path, distA_names, severities_A, resize)
        gen_distortion_img(img_path, imgB_path, distB_names, severities_B, resize)

        meta = {
            "id": f"{img_name}_{idx}",
            "img_ref": img_path,
            "img_lq_A": {
                "img_path": imgA_path,
                "distortion_classes": distA_classes,
                "distortion_names": distA_names,
                "severities": severities_A,
            },
            "img_lq_B": {
                "img_path": imgB_path,
                "distortion_classes": distB_classes,
                "distortion_names": distB_names,
                "severities": severities_B,
            },
        }
        with open(save_json, "w") as fw:
            fw.write(json.dumps(meta, indent=4))

        print(meta)
