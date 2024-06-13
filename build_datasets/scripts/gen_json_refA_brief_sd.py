import argparse
import glob
import json
import os
import random

from scripts.constants_refA_qr import brief_good, qr_list, single_q_tail

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--split_json", type=str, required=True)
parser.add_argument("--meta_dir", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--training", action="store_true")
# diff seeds for test sampling: refA_sd: 131+1, refA_md: 131+2, refAB_sd: 131+3, refAB_md: 131+4
parser.add_argument("--seed", type=int, default=132)


def generate_conversations_brief(meta):
    conversations = []
    idx = random.randint(0, len(qr_list) - 1)
    if random.random() <= p_number:
        q_str = qr_list[idx][0][1]
        r_str = qr_list[idx][1].replace("{ONE}", "ONE")
    else:
        q_str = qr_list[idx][0][0]
        r_str = qr_list[idx][1].replace("{ONE} ", "")

    q_dict = {"from": "human", "value": q_str}
    conversations.append(q_dict)
    degradation = degradation_rename(meta["distortion_name"])
    if degradation:
        r_str = r_str.replace("{}", degradation)
    else:
        r_str = brief_good
    r_dict = {"from": "gpt", "value": r_str}
    conversations.append(r_dict)
    return conversations


def generate_conversations_single(meta):
    conversations = []
    idx = random.randint(0, len(qr_list) - 1)
    if random.random() <= p_number:
        q_str = qr_list[idx][0][1]
    else:
        q_str = qr_list[idx][0][0]

    q_dict = {"from": "human", "value": q_str + single_q_tail}
    conversations.append(q_dict)
    degradation = degradation_rename(meta["distortion_name"])
    if degradation:
        degradation = degradation[0].upper() + degradation[1:]
    else:
        degradation = "None"
    r_dict = {"from": "gpt", "value": degradation}
    conversations.append(r_dict)
    return conversations


def degradation_rename(degradation):
    if not degradation:  # no distortion
        return None
    if "noise" in degradation:
        return "noise"
    elif "blur" in degradation:
        return "blur"
    elif "compression" in degradation:
        return "compression"
    elif "oversharpen" in degradation:
        return "oversharpening"
    elif "pixelate" in degradation:
        return "pixelation"
    elif "quantization" in degradation:
        return "color quantization"
    elif "saturate_weaken" in degradation:
        return "insufficient saturation"
    elif "saturate_strengthen" in degradation:
        return "overly high saturation"
    elif "contrast_weaken" in degradation:
        return "insufficient contrast"
    elif "contrast_strengthen" in degradation:
        return "overly high contrast"
    elif "brightness_darken" in degradation:
        return "insufficient brightness"
    elif "brightness_brighten" in degradation:
        return "overly high brightness"
    else:
        raise NotImplementedError(f"{degradation} Not implemented yet")


def seed_everything(seed=131):
    random.seed(seed**2)


if __name__ == "__main__":
    args = parser.parse_args()
    meta_dir = args.meta_dir
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    training = args.training
    seed_everything(args.seed)

    p_single = 0.5 if training else 1  # test -> all single
    p_number = 0.5 if training else 1  # test -> all number
    meta_list = sorted(glob.glob(os.path.join(meta_dir, "*.json")))

    split_json = args.split_json
    with open(split_json) as fr:
        split_dict = json.load(fr)
    img_names = split_dict["train"] if training else split_dict["test"]
    if not training:
        num_test = 5000
        img_names = random.sample(img_names, num_test)

    num_brief = 0
    num_single = 0
    metas_save = []
    for idx, meta_file in enumerate(meta_list):
        print(idx)
        with open(meta_file) as fr:
            meta = json.load(fr)
        if not os.path.basename(meta["img_ref"]) in img_names:
            continue

        meta["id"] = os.path.splitext(os.path.basename(meta["img_lq"]))[0]
        meta["image_ref"] = meta["img_ref"]
        meta["image_A"] = meta["img_lq"]
        meta["image_B"] = None
        meta["task_type"] = "quality_single_A"
        del meta["img_ref"]
        del meta["img_lq"]

        if random.random() <= p_single:
            num_single += 1
            conversations = generate_conversations_single(meta)
        else:
            num_brief += 1
            conversations = generate_conversations_brief(meta)
        meta["conversations"] = conversations

        if not training:
            meta["query"] = meta["conversations"][0]["value"]
            meta["answer"] = meta["conversations"][1]["value"]
            del meta["conversations"]

        metas_save.append(meta)

    print(f"Brief: {num_brief}, Single: {num_single}")
    with open(save_path, "w") as fw:
        print(f"{len(metas_save)} samples saved to {save_path}")
        fw.write(json.dumps(metas_save, indent=4))
