import argparse
import json
import os
import random

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--json_original", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--training", action="store_true")
parser.add_argument("--seed", type=int, default=131)


single_q_tail = " Answer the question using a single word or phrase."

brief_qr_list = [
    [
        "Which image do you believe has better overall quality: Image A or Image B?",
        "I believe Image {} has better overall quality.",
    ],
    [
        "Determine which image exhibits higher quality between Image A and Image B.",
        "In my assessment, Image {} exhibits higher quality.",
    ],
    [
        "Compare the general quality of Image A and Image B, and state your preference.",
        "My preference leans towards Image {} to have better general quality.",
    ],
    [
        "In your opinion, which image demonstrates superior quality: Image A or Image B?",
        "In my opinion, Image {} demonstrates superior quality.",
    ],
    [
        "Which of the two images, Image A or Image B, do you consider to be of better quality?",
        "I consider Image {} to be of better quality.",
    ],
    [
        "Evaluate the quality of Image A and Image B, and decide which one is superior.",
        "I conclude that Image {} is superior.",
    ],
    [
        "Between Image A and Image B, which image do you think has better quality overall?",
        "I think Image {} has better quality overall. ",
    ],
    [
        "Determine which image, Image A or Image B, you perceive to have better quality.",
        "I determine that Image {} has better quality.",
    ],
    [
        "Assess the quality of Image A and Image B, and choose the one you believe is superior.",
        "I choose Image {} to be superior in terms of quality.",
    ],
    [
        "Which image stands out to you as having better quality: Image A or Image B?",
        "Image {} stands out as the superior choice in terms of quality.",
    ],
    [
        "Can you compare the quality of Image A and Image B and decide which one is better?",
        "I find Image {} to be better after comparing the quality of both.",
    ],
    [
        "Decide which image, Image A or Image B, you think possesses higher quality.",
        "I decide that Image {} possesses higher quality.",
    ],
    [
        "Evaluate Image A and Image B, and select the one that you feel has better quality.",
        "Upon evaluation, I select Image {} as the one with better quality.",
    ],
    [
        "Which of the two images, Image A or Image B, appears to have superior quality to you?",
        "To me, Image {} appears to have superior quality.",
    ],
    [
        "Compare the quality of Image A and Image B, and determine which one you prefer.",
        "My preference leans towards Image {} after comparing the quality.",
    ],
    [
        "Make a judgment on which image, Image A or Image B, you consider to be of better quality.",
        "I consider Image {} to be of better quality.",
    ],
    [
        "Between Image A and Image B, which image do you perceive to have better quality overall?",
        "I perceive Image {} to have better quality overall.",
    ],
    [
        "Assess the quality of Image A and Image B, and indicate which one you find to be better.",
        "I find Image {} emerges as the better option with superior quality.",
    ],
    [
        "Which image, Image A or Image B, do you think displays better quality when compared?",
        "When compared, Image {} displays better quality.",
    ],
    [
        "Differentiate between Image A and Image B in terms of overall quality and decide which one is superior.",
        "Image {} differentiates itself with superior quality.",
    ],
]


def generate_conversations_brief(meta):
    better = "A" if meta["A_better_B"] else "B"
    idx = random.randint(0, len(brief_qr_list) - 1)
    conversations = [
        {"from": "human", "value": brief_qr_list[idx][0]},
        {"from": "gpt", "value": brief_qr_list[idx][1].replace("{}", better)},
    ]
    return conversations


def generate_conversations_single(meta):
    better = "A" if meta["A_better_B"] else "B"
    idx = random.randint(0, len(brief_qr_list) - 1)
    conversations = [
        {"from": "human", "value": brief_qr_list[idx][0] + single_q_tail},
        {"from": "gpt", "value": f"Image {better}"},
    ]
    return conversations


def seed_everything(seed=131):
    random.seed(seed**2)


if __name__ == "__main__":
    args = parser.parse_args()
    json_original = args.json_original
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    seed_everything(args.seed)

    training = args.training
    p_single = 0.5 if training else 1  # test -> all single
    with open(json_original) as fr:
        metas = json.load(fr)

    num_brief = 0
    num_single = 0
    metas_save = []
    for meta in metas:
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
