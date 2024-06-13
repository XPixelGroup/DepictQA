import argparse
import logging
import os
import pprint
import random
import time

import deepspeed
import numpy as np
import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm
from transformers.deepspeed import HfDeepSpeedConfig

from datasets import load_trainset
from model import build_agent

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parser_args():
    parser = argparse.ArgumentParser(description="train parameters for DepictQA")
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    return args


def initialize_distributed(args):
    args["master_ip"] = os.getenv("MASTER_ADDR", "localhost")
    args["master_port"] = os.getenv("MASTER_PORT", "6000")
    args["world_size"] = int(os.getenv("WORLD_SIZE", "1"))
    args["local_rank"] = int(os.getenv("RANK", "0")) % torch.cuda.device_count()
    device = args["local_rank"] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    deepspeed.init_distributed(dist_backend="nccl")


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(args):
    start_time = time.time()
    initialize_distributed(args)
    set_random_seed(args["seed"])
    num_epochs = args.train["epochs"]
    save_dir = args.train["save_dir"]
    log_dir = args.train["log_dir"]
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    args.deepspeed["train_batch_size"] = (
        args["world_size"]
        * args.deepspeed["train_micro_batch_size_per_gpu"]
        * args.deepspeed["gradient_accumulation_steps"]
    )
    dschf = HfDeepSpeedConfig(args["deepspeed"])
    args["dschf"] = dschf

    time_str = time.strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        format="%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s",
        level=logging.INFO,
        filename=os.path.join(log_dir, f"train_{time_str}.log"),
        filemode="w",
    )

    dataloader = load_trainset(args)
    length = (
        num_epochs
        * len(dataloader.dataset)
        // args["world_size"]
        // dschf.config["train_micro_batch_size_per_gpu"]
    )
    total_steps = (
        num_epochs * len(dataloader.dataset) // dschf.config["train_batch_size"]
    )
    args.train["total_steps"] = total_steps
    agent = build_agent(args, training=True)
    torch.distributed.barrier()

    with open(os.path.join(log_dir, "training_args.yaml"), "w") as fw:
        yaml.dump(args, fw)
    logging.info("args: {}".format(pprint.pformat(args)))

    # train begin
    pbar = tqdm(total=length)  # maximum total number
    step = 0
    for epoch in tqdm(range(num_epochs)):
        for batch in dataloader:
            agent.train_model(batch, step=step, pbar=pbar)
            step += 1
            if args.train["save_freq_step"]:
                if step % args.train["save_freq_step"] == 0:
                    agent.save_model(save_dir, epoch + 1, step)
        if epoch % args.train["save_freq_epoch"] == 0:
            agent.save_model(save_dir, epoch + 1)

    # train end
    torch.distributed.barrier()
    agent.save_model(save_dir)
    logging.info(f"[!] Traing done. Total time: {time.time() - start_time}")


if __name__ == "__main__":
    args = parser_args()
    with open(args.cfg, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    args = vars(args)
    cfg.update(args)
    main(cfg)
