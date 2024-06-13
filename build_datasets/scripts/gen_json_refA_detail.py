import argparse
import glob
import json
import os
import random

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--meta_dir", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=131)


q_list = [
    "Could you assess the overall quality of the image and elaborate on your evaluation?",
    "How would you rate the image's quality, and what factors contribute to your assessment?",
    "Can you provide a detailed evaluation of the image's quality?",
    "Please evaluate the image's quality and provide your reasons.",
    "How do you perceive the quality of the image, and what aspects influence your judgment?",
    "Offer an assessment of the image's quality, highlighting any strengths or weaknesses.",
    "What is your opinion on the quality of the image? Explain your viewpoint.",
    "Assess the quality of the image with detailed reasons.",
    "How does the image's quality impact its overall effectiveness or appeal?",
    "Evaluate the image's quality and justify your evaluation.",
    "How about the overall quality of the image, and why?",
    "Provide a thorough evaluation of the image's quality.",
    "Examine the image's quality by considering factors influencing its clarity.",
    "Analyze the image's quality, and detail your findings.",
    "Provide a comprehensive assessment of the image's quality, including both strengths and areas for improvement.",
    "Assess the image's quality from a professional standpoint.",
    "Evaluate the image's clarity and explain how it contributes to the overall quality.",
    "How would you rate the overall quality of the image, and why?",
    "What is your opinion on the image's quality? Elaborate on your evaluation.",
    "Evaluate the quality of the image and provide a comprehensive explanation.",
]


def seed_everything(seed=131):
    random.seed(seed**2)


if __name__ == "__main__":
    args = parser.parse_args()
    meta_dir = args.meta_dir
    save_path = args.save_path
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    seed_everything(args.seed)

    meta_list = sorted(glob.glob(os.path.join(meta_dir, "*.json")))

    metas = []
    for meta_file in meta_list:
        with open(meta_file) as fr:
            meta = json.load(fr)
        if "sorry" in meta["text"]:
            print(meta_file)
            continue

        meta["image_ref"] = meta["img_ref"]
        meta["image_A"] = meta["img_lq"]
        meta["image_B"] = None
        meta["id"] = os.path.splitext(os.path.basename(meta["img_lq"]))[0]
        del meta["img_ref"]
        del meta["img_lq"]

        meta["task_type"] = "quality_single_A"
        meta["conversations"] = [
            {"from": "human", "value": random.choice(q_list)},
            {"from": "gpt", "value": meta["text"]},
        ]
        del meta["text"]
        metas.append(meta)

    with open(save_path, "w") as fw:
        print(f"{len(metas)} samples saved to {save_path}")
        fw.write(json.dumps(metas, indent=4))
