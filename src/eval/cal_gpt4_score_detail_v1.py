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
    "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above. "
    + "The user asks the question on assessing the quality of an image. "
    + "Please rate the helpfulness, relevance, accuracy, level of details of their responses. "
    + "Each assistant receives an overall score on a scale of 0 to 10, where a higher score indicates better overall performance. "
    + "Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. "
    + "The two scores are separated by a space. "
    + "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.\n",
}


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        if len(sp) == 2:
            return [float(sp[0]), float(sp[1])]
        else:
            print("error", review)
            return [-1, -1]
    except Exception as e:
        print(e)
        print("error", review)
        return [-1, -1]


def gen_res_from_gpt(content):
    openai.api_key = ""
    gpt_model = "gpt-4-0314"

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


def gen_context(gt):
    res = ""

    gt_texture = gt["texture"]
    res += f"texture type: {gt_texture['texture_type']}\n"
    res += f"texture damage: {gt_texture['texture_damage']}\n"

    gt_distortion = gt["distortion"]
    res += f"brightness distortion: {gt_distortion['bright']}\n"
    res += f"color distortion: {gt_distortion['color']}\n"
    res += f"noise: {gt_distortion['noise']}\n"
    res += f"artifacts: {gt_distortion['artifact']}\n"
    res += f"blurriness: {gt_distortion['blur']}\n\n"

    res += f"overall: {gt['overall']}"

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
    scores_list = []
    save_path = args.save_path
    if os.path.isfile(save_path):
        with open(save_path) as fr:
            for line in fr:
                meta = json.loads(line)
                dist_ids.append(meta["id"])
                scores_list.append(meta["scores"])

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

        context = gen_context(gt)
        question = gt["query"]
        answer_gt = gt["answer"]
        answer_pred = pred["text"]

        content = (
            f"[Context]\n{context}\n\n"
            f"[Question]\n{question}\n\n"
            f"[Assistant 1]\n{answer_gt}\n\n[End of Assistant 1]\n\n"
            f"[Assistant 2]\n{answer_pred}\n\n[End of Assistant 2]\n\n"
            f"[System]\n{DEFAULT_SETTINGS['prompt']}\n\n"
        )

        cur_js = {
            "index": idx,
            "id": meta_id,
            "answer_gt": answer_gt,
            "answer_pred": answer_pred,
        }

        review = gen_res_from_gpt(content)
        scores = parse_score(review)
        scores_list.append(scores)
        cur_js["content"] = review
        cur_js["scores"] = scores
        print(cur_js)
        review_file.write(json.dumps(cur_js) + "\n")
        review_file.flush()

    review_file.close()

    scores = [min(_[1] / _[0], 1.0) for _ in scores_list if _[0] >= 0]
    score = sum(scores) / len(scores)
    print(f"GPT4 Score: {score}")
