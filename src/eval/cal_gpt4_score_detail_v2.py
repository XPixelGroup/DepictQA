import argparse
import json
import os

import openai

parser = argparse.ArgumentParser(description="evaluation parameters for DepictQA")
parser.add_argument("--pred_path", type=str, required=True)
parser.add_argument("--gt_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)


DEFAULT_SETTINGS = {
    "system_prompt": "You are a helpful and precise assistant for checking the quality of the answer.",
    "prompt": "We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. "
    + "The user asks the question on assessing the image quality. "
    + "The ground truth is given for your evaluation. "
    + "Please rate the consistency between the assistant's response and the ground truth. "
    + "Pay attention to the distortion analyses. "
    + "The assistant receives an overall score on a scale of 0 to 10, where a higher score indicates better performance. "
    + "Please first output a single line containing ONLY ONE INT NUMBER indicating the score of the assistant. "
    + "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.\n",
}


def parse_score(review):
    try:
        score = review.split("\n")[0].strip()
        try:
            return float(score)
        except:
            print("error", review)
            return -1
    except Exception as e:
        print(e)
        print("error", review)
        return -1


def gen_res_from_gpt(content):
    openai.api_key = ""
    gpt_model = "gpt-4-turbo"

    response = openai.ChatCompletion.create(
        model=gpt_model,
        messages=[
            {
                "role": "system",
                "content": DEFAULT_SETTINGS["system_prompt"],
            },
            {
                "role": "user",
                "content": content,
            },
        ],
        temperature=0.0,
    )
    res = response["choices"][0]["message"]["content"]
    return res


if __name__ == "__main__":
    args = parser.parse_args()

    # load predict results
    pred_path = args.pred_path
    if pred_path.endswith(".jsonl"):
        pred_metas = []
        with open(pred_path) as fr:
            for line in fr:
                pred_meta = json.loads(line)
                pred_metas.append(pred_meta)
    else:
        assert pred_path.endswith(".json")
        with open(pred_path) as fr:
            pred_metas = json.load(fr)

    # load gt results
    with open(args.gt_path) as fr:
        gt_metas = json.load(fr)

    for pred, gt in zip(pred_metas, gt_metas):
        assert pred["id"] == gt["id"]

    dist_ids = []
    scores = []
    save_path = args.save_path
    if os.path.isfile(save_path):
        with open(save_path) as fr:
            for line in fr:
                meta = json.loads(line)
                dist_ids.append(meta["id"])
                scores.append(meta["score"])

    idx_end = 200
    review_file = open(save_path, "a")
    for idx, (gt, pred) in enumerate(zip(gt_metas[:idx_end], pred_metas[:idx_end])):
        assert gt["id"] == pred["id"]
        meta_id = pred["id"]
        print("=" * 100)
        print(f"Handling {meta_id}")
        if meta_id in dist_ids:
            print(f"Skipping {meta_id} as we already have it.")
            continue

        question = gt["query"]
        answer_gt = gt["answer"]
        answer_pred = pred["text"]

        content = (
            f"[Question]\n{question}\n\n"
            f"[Ground Truth]\n{answer_gt}\n\n[End of Ground Truth]\n\n"
            f"[Assistant]\n{answer_pred}\n\n[End of Assistant]\n\n"
            f"[System]\n{DEFAULT_SETTINGS['prompt']}\n\n"
        )

        cur_js = {
            "index": idx,
            "id": meta_id,
            "answer_gt": answer_gt,
            "answer_pred": answer_pred,
        }

        review = gen_res_from_gpt(content)
        score = parse_score(review)
        scores.append(score)
        cur_js["content"] = review
        cur_js["score"] = score
        print(cur_js)
        review_file.write(json.dumps(cur_js) + "\n")
        review_file.flush()

    review_file.close()

    scores = [_ for _ in scores if _ >= 0]
    score = sum(scores) / len(scores)
    print(f"GPT4 Score: {score}")
