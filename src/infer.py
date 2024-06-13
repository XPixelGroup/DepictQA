import argparse
import json
import logging
import os
import pprint

import torch
import yaml
from bigmodelvis import Visualization
from easydict import EasyDict
from tqdm import tqdm

from datasets import load_valset
from model.depictqa import DepictQA


def parse_args():
    parser = argparse.ArgumentParser(description="infer parameters for DepictQA")
    parser.add_argument("--cfg", type=str, default="config.yaml")
    # infer cfgs, overwrite cfg.data.infer if set
    parser.add_argument("--meta_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()
    return args


def main(args):
    logging.info("args: {}".format(pprint.pformat(args)))
    assert os.path.exists(
        args.model["vision_encoder_path"]
    ), "vision_encoder_path not exist!"
    assert os.path.exists(args.model["llm_path"]), "llm_path not exist!"
    assert os.path.exists(args.model["delta_path"]), "delta_path not exist!"

    # load model
    model = DepictQA(args, training=False)
    delta_ckpt = torch.load(args.model["delta_path"], map_location=torch.device("cpu"))
    model.load_state_dict(delta_ckpt, strict=False)
    model = model.eval().half().cuda()
    Visualization(model).structure_graph()
    logging.info(f"[!] init the LLM over ...")

    # load data
    dataloader = load_valset(args)
    answer_path = os.path.join(
        args.infer["answer_dir"],
        args.data.infer["task_name"] + "_" + args.data.infer["dataset_name"] + ".json",
    )
    os.makedirs(os.path.dirname(answer_path), exist_ok=True)

    answers = []
    with open(os.path.splitext(answer_path)[0] + ".jsonl", "w") as fw:
        for meta in tqdm(dataloader):
            texts, output_ids, probs, confidences = model.generate(
                {
                    "query": meta["query"],
                    "img_path": meta["img_path"],
                    "img_A_path": meta["img_A_path"],
                    "img_B_path": meta["img_B_path"],
                    "temperature": args.infer["temperature"],
                    "top_p": args.infer["top_p"],
                    "max_new_tokens": args.infer["max_new_tokens"],
                    "task_type": args.data.infer["task_name"],
                    "output_prob_id": args.infer["output_prob_id"],
                    "output_confidence": args.infer["output_confidence"],
                    "sentence_model": args.infer["sentence_model"],
                }
            )

            for meta_id, text, output_id, prob, confidence in zip(
                meta["id"], texts, output_ids, probs, confidences
            ):
                output_id = (
                    [int(_) for _ in output_id.cpu().numpy()]
                    if output_id is not None
                    else output_id
                )
                prob = (
                    [float(_) for _ in prob.cpu().numpy()] if prob is not None else prob
                )
                answer = {
                    "id": meta_id,
                    "text": text,
                    "output_id": output_id,
                    "prob": prob,
                    "confidence": confidence,
                }
                answers.append(answer)
                fw.write(json.dumps(answer) + "\n")
                fw.flush()

    with open(answer_path, "w") as fw:
        fw.write(json.dumps(answers, indent=4))


if __name__ == "__main__":
    args = parse_args()
    with open(args.cfg, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    args = vars(args)
    args_filter = {}
    for key in args.keys():
        if key != "cfg" and args[key] is not None:
            args_filter[key] = args[key]
    # command line has higher priority
    cfg.data.infer.update(args_filter)
    main(cfg)
