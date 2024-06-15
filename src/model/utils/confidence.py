import copy

import numpy as np
from sentence_transformers import util

single_q_tail = " Answer the question using a single word or phrase."
# If a question contains one of the following str, take it as the question with single prompt
single_q_tail_list = [
    "single word",
    "one word",
    "a word",
    "single phrase",
    "one phrase",
    "a phrase",
]


def cal_confidence_batch(model, inputs, output_texts, output_ids, output_probs):
    batch_size = len(output_texts)
    confidences = []
    for idx in range(batch_size):
        # online demo (batch_size = 1) not support deepcopy
        inputs_copy = inputs if batch_size == 1 else copy.deepcopy(inputs)
        inputs_copy["query"] = [inputs["query"][idx]]
        inputs_copy["img_path"] = [inputs["img_path"][idx]]
        inputs_copy["img_A_path"] = [inputs["img_A_path"][idx]]
        inputs_copy["img_B_path"] = [inputs["img_B_path"][idx]]
        output_text = output_texts[idx]
        output_id = output_ids[idx]
        output_prob = output_probs[idx]
        confidence = cal_confidence(
            model, inputs_copy, output_text, output_id, output_prob
        )
        confidences.append(confidence)
    return confidences


def cal_confidence(model, inputs, text, output_id, prob):
    words = text.split(" ")
    # in training, max len of brief = 19, min len of detail = 24
    if len(words) < 22:  # brief
        try:
            if "quality_single" in inputs["task_type"]:
                confidence = cal_confidence_single_brief(model, inputs, output_id, prob)
            elif "quality_compare" in inputs["task_type"]:
                confidence = cal_confidence_compare_brief(
                    model, inputs, output_id, prob
                )
            else:
                raise NotImplementedError
            return confidence
        except:
            confidence = None
    elif model.sentence_model:  # detail
        confidence = cal_confidence_detail(model, text, output_id, prob)
    else:
        confidence = None
    return confidence


def cal_confidence_single_brief(model, inputs, output_id, prob):
    for tail in single_q_tail_list:
        if tail in inputs["query"][0]:
            confidence = _cal_confidence_single_brief(output_id, prob)
            return confidence
    inputs["query"][0] = inputs["query"][0] + single_q_tail
    _, _, _, confidence = model.generate(inputs)
    return confidence[0]


def _cal_confidence_single_brief(output_id, prob):
    output_id = [int(_) for _ in output_id.cpu().numpy()]
    prob = [float(_) for _ in prob.cpu().numpy()]
    output_id = output_id[1:]  # remove start token
    if 0 in output_id:
        idx = output_id.index(0)
        assert output_id[idx:] == [0] * (len(output_id) - idx)
        output_id = output_id[:idx]
        prob = prob[:idx]

    idx = 0
    while True:
        # remove "\n " ([13, 29871, 2])
        if output_id[idx : idx + 3] == [13, 29871, 2] or idx >= len(output_id):
            break
        idx += 1
    output_id = output_id[:idx]
    prob = prob[:idx]

    if 322 in output_id:  # remove "and" (322)
        idx_and = output_id.index(322)
        del output_id[idx_and]
        del prob[idx_and]
    assert len(output_id) == len(prob)
    confidence = sum(prob) / len(prob)
    return confidence


def cal_confidence_compare_brief(model, inputs, output_id, prob):
    for tail in single_q_tail_list:
        if tail in inputs["query"][0]:
            confidence = _cal_confidence_compare_brief(output_id, prob)
            return confidence
    inputs["query"][0] = inputs["query"][0] + single_q_tail
    _, _, _, confidence = model.generate(inputs)
    return confidence[0]


def _cal_confidence_compare_brief(output_id, prob):
    output_id = [int(_) for _ in output_id.cpu().numpy()]
    prob = [float(_) for _ in prob.cpu().numpy()]
    output_id = output_id[1:]  # remove start token
    if 0 in output_id:
        idx = output_id.index(0)
        assert output_id[idx:] == [0] * (len(output_id) - idx)
        output_id = output_id[:idx]
        prob = prob[:idx]

    idx = 0
    while True:
        # [7084, 319]: "Image A", [7084, 350]: "Image B"
        if output_id[idx : idx + 2] in [[7084, 319], [7084, 350]]:
            break
        idx += 1

    confidence = prob[idx + 1]  # Take the prob of "A" and "B" as confidence
    return confidence


def cal_confidence_detail(model, text, output_id, prob):
    num_max_id = 30
    embed_org = model.sentence_model.encode(text)

    output_id = [int(_) for _ in output_id.cpu().numpy()]
    prob = [float(_) for _ in prob.cpu().numpy()]
    output_id = output_id[1:]  # remove start token
    if 0 in output_id:
        idx = output_id.index(0)
        assert output_id[idx:] == [0] * (len(output_id) - idx)
        output_id = output_id[:idx]
        prob = prob[:idx]
    assert model.tokenizer.decode(output_id).startswith(text)

    # calculate importance weights for all tokens
    texts_del = []
    for idx in range(len(output_id)):
        output_id_del = output_id[:idx] + output_id[idx + 1 :]
        text_del = model.tokenizer.decode(output_id_del, skip_special_tokens=True)
        texts_del.append(text_del)
    embed_del = model.sentence_model.encode(texts_del)
    weight_list = [
        float(_) for _ in (1 - util.cos_sim(embed_org, embed_del)).squeeze(0).numpy()
    ]

    # select important tokens & calculate the weighted confidence
    idx_sort = sorted(
        range(len(weight_list)), key=lambda x: weight_list[x], reverse=True
    )[:num_max_id]
    output_id_sort = [output_id[idx] for idx in idx_sort]
    prob_sort = np.array([prob[idx] for idx in idx_sort])
    weight_sort = np.array(sorted(weight_list, reverse=True)[:num_max_id])
    weight_norm = weight_sort / weight_sort.sum()
    confidence = (weight_norm * prob_sort).sum()
    return confidence
