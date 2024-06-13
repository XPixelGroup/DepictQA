import copy

import numpy as np
import torch

np.random.seed(131)

import torch.nn as nn
from sentence_transformers import SentenceTransformer, util
from transformers import LlamaTokenizer

text = "The images depict an indoor corridor with tiled flooring, walls, and a ceiling. In the evaluated image, saturation has been moderately increased, resulting in unnaturally vibrant and intense colors that reduce the realism and detail of the scene. The distortion affects the image by making it less natural and potentially straining to the eyes. Overall, the quality of the evaluated image is diminished compared to the reference due to the exaggerated color saturation, which detracts from the visual comfort and authenticity of the environment."
tokenizer_path = "/opt/data/private/142/Model_zoo/LLM/vicuna/vicuna-7b-v1.5"
sentence_path = "/opt/data/private/142/Model_zoo/sentence_transformers/all-MiniLM-L6-v2"

if __name__ == "__main__":
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    sentence_model = SentenceTransformer(sentence_path, trust_remote_code=True)
    model = nn.Module()
    model.tokenizer = tokenizer
    model.sentence_model = sentence_model

    output_id = torch.tensor(tokenizer.encode(text))
    prob = torch.tensor(np.random.rand(len(output_id) - 1))

    num_max_id = 50
    emb_org = model.sentence_model.encode(text)

    output_id = [int(_) for _ in output_id.cpu().numpy()]
    prob = [float(_) for _ in prob.cpu().numpy()]
    output_id = output_id[1:]  # remove start token
    if 0 in output_id:
        idx = output_id.index(0)
        assert output_id[idx:] == [0] * (len(output_id) - idx)
        output_id = output_id[:idx]
        prob = prob[:idx]
    assert model.tokenizer.decode(output_id) == text

    # calculate importance weights for all tokens
    texts_del = []
    for idx in range(len(output_id)):
        output_id_del = output_id[:idx] + output_id[idx + 1 :]
        text_del = model.tokenizer.decode(output_id_del, skip_special_tokens=True)
        texts_del.append(text_del)
    emb_del = model.sentence_model.encode(texts_del)
    weight_list = [
        float(_) for _ in (1 - util.cos_sim(emb_org, emb_del)).squeeze(0).numpy()
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
    print(confidence)

    ########################################
    # verify the remove of "dist"
    # remove dist (1320)
    idx_dist = output_id.index(1320)
    output_id_dist = copy.deepcopy(output_id)
    del output_id_dist[idx_dist]
    text_dist = tokenizer.decode(output_id_dist, skip_special_tokens=True)
    emb_dist = sentence_model.encode(text_dist)
    weight_dist = float(1 - util.cos_sim(emb_org, emb_dist))
    assert round(weight_dist, 5) == round(float(weight_sort[0]), 5)

    ########################################
    # verify the remove of "cor"
    # remove cor (1034)
    idx_cor = output_id.index(1034)
    output_id_cor = copy.deepcopy(output_id)
    del output_id_cor[idx_cor]
    text_cor = tokenizer.decode(output_id_cor, skip_special_tokens=True)
    emb_cor = sentence_model.encode(text_cor)
    weight_cor = float(1 - util.cos_sim(emb_org, emb_cor))
    assert round(weight_cor, 5) == round(float(weight_sort[1]), 5)

    # calculate importance weights for all tokens
    weight_list_new = []
    for idx in range(len(output_id)):
        output_id_del = output_id[:idx] + output_id[idx + 1 :]
        text_del = model.tokenizer.decode(output_id_del, skip_special_tokens=True)
        emb_del = model.sentence_model.encode(text_del)
        weight = float(1 - util.cos_sim(emb_org, emb_del))
        weight_list_new.append(weight)
    for weight, weight_new in zip(weight_list, weight_list_new):
        assert round(weight, 5) == round(weight_new, 5)
