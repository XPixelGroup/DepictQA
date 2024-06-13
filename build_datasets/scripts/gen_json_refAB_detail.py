import argparse
import glob
import json
import os
import random

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--meta_dir", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--training", action="store_true")
parser.add_argument("--seed", type=int, default=131)


q_list = [
    "Compare the overall quality of Image A with Image B and provide a comprehensive explanation.",
    "Which image has better visual quality, Image A or Image B? Can you explain the comparison results?",
    "Evaluate the general visual appeal and quality of both Image A and Image B, and elaborate on which one excels.",
    "Discuss the overall impression and quality of Image A versus Image B, and justify your assessment.",
    "Compare the overall quality between Image A and Image B, and justify your comparison results.",
    "Assess the overall visual quality of Image A and Image B, discussing which one delivers a more compelling visual quality.",
    "Which image demonstrates higher overall quality, Image A or Image B? Please provide detailed reasoning for your evaluation.",
    "Analyze the overall quality of both Image A and Image B, and explain which image stands out.",
    "Compare the perceived quality of Image A with Image B, providing insights into their respective strengths and weaknesses.",
    "Discuss the visual quality of Image A and Image B, and elaborate on which one appears more appealing.",
    "Can you evaluate the overall quality in both Image A and Image B, and explain which one is superior?",
    "Compare the overall visual impact and impression of Image A versus Image B, and justify your assessment of their quality.",
    "Which image exhibits higher overall quality: Image A or Image B? Please explain your reasoning.",
    "Evaluate the visual quality in Image A and Image B, providing insights into their comparative strengths.",
    "Compare the overall quality between Image A and Image B, and discuss which one appears more appealing.",
    "Assess the visual quality of both Image A and Image B, and explain which one is better.",
    "Which image demonstrates superior quality: Image A or Image B? Please elaborate on your evaluation.",
    "Discuss the overall impression of Image A versus Image B, and justify your assessment of their comparative quality.",
    "Compare the visual quality of Image A with Image B, providing detailed insights into their respective strengths and weaknesses.",
    "Evaluate the overall quality of Image A and Image B, and explain which one has higher quality.",
]


def seed_everything(seed=131):
    random.seed(seed**2)


if __name__ == "__main__":
    args = parser.parse_args()
    meta_dir = args.meta_dir
    save_path = args.save_path
    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
    training = args.training
    seed_everything(args.seed)

    meta_list = sorted(glob.glob(os.path.join(meta_dir, "*.json")))

    metas = []
    for meta_file in meta_list:
        with open(meta_file) as fr:
            meta = json.load(fr)
        if training:
            if "sorry" in meta["text"]:
                print(meta_file)
                continue

        meta["id"] = os.path.splitext(os.path.basename(meta_file))[0]
        meta["image_ref"] = meta["img_ref"]
        meta["image_A"] = meta["img_lq_A"]["img_path"]
        meta["image_B"] = meta["img_lq_B"]["img_path"]
        del meta["img_ref"]
        del meta["img_lq_A"]["img_path"]
        del meta["img_lq_B"]["img_path"]

        meta["distortions_A"] = meta["img_lq_A"]
        meta["distortions_B"] = meta["img_lq_B"]
        del meta["img_lq_A"]
        del meta["img_lq_B"]

        if training:
            meta["conversations"] = [
                {"from": "human", "value": random.choice(q_list)},
                {"from": "gpt", "value": meta["text"]},
            ]
            del meta["text"]
        else:
            meta["query"] = random.choice(q_list)
        meta["task_type"] = "quality_compare"
        metas.append(meta)

    with open(save_path, "w") as fw:
        print(f"{len(metas)} samples saved to {save_path}")
        fw.write(json.dumps(metas, indent=4))
