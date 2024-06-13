import datetime
import logging
import os
import types
from collections import OrderedDict

import deepspeed
import torch
from torch.utils.tensorboard import SummaryWriter


class DeepSpeedAgent:
    def __init__(self, model, args):
        super(DeepSpeedAgent, self).__init__()
        self.args = args
        self.model = model

        self.writer = SummaryWriter(args.train["log_dir"])
        delta_path = args.model["delta_path"]
        if os.path.exists(delta_path):
            delta_ckpt = torch.load(delta_path, map_location=torch.device("cpu"))
            self.model.load_state_dict(delta_ckpt, strict=False)
            logging.info(f"[!] Load pretrained delta ckpt from {delta_path}")
        else:
            logging.info(f"[!] Train from scratch, delta_path ({delta_path}) not exist")

        # load config parameters of deepspeed
        ds_params = args["deepspeed"]
        ds_params["scheduler"]["params"]["total_num_steps"] = self.args.train[
            "total_steps"
        ]
        ds_params["scheduler"]["params"]["warmup_num_steps"] = max(
            10, int(self.args.train["total_steps"] * self.args.train["warmup_rate"])
        )
        self.ds_engine, self.optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            model_parameters=self.model.parameters(),
            config_params=ds_params,
            dist_init_required=True,
            args=types.SimpleNamespace(**args),
        )

    def train_model(self, batch, step=0, pbar=None):
        self.ds_engine.module.train()
        loss, acc = self.ds_engine(batch)

        self.ds_engine.backward(loss)
        self.ds_engine.step()
        pbar.set_description(
            f"[!] loss: {round(loss.item(), 4)}; acc: {round(acc * 100, 2)}"
        )
        pbar.update(1)
        if self.args["local_rank"] == 0 and step % self.args.train["log_step"] == 0:
            elapsed = pbar.format_dict["elapsed"]
            rate = pbar.format_dict["rate"]
            remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
            remaining = str(datetime.timedelta(seconds=remaining))
            self.writer.add_scalar("train/loss", loss.item(), step)
            self.writer.add_scalar("train/acc", acc * 100, step)
            logging.info(
                f"[!] progress: {round(pbar.n / pbar.total, 4)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; acc: {round(acc * 100, 2)}"
            )
        acc *= 100
        return acc

    def save_model(self, save_dir, epoch=None, step=None):
        # only save trainable parameters
        trainable_params = [
            k for (k, v) in self.ds_engine.module.named_parameters() if v.requires_grad
        ]
        # state_dict on rank 0
        # NOTE: state_dict is still none in other processes
        state_dict = None
        if self.ds_engine.zero_optimization_partition_weights():
            if self.ds_engine.zero_gather_16bit_weights_on_model_save():
                # consolidation is expensive in time and memory and therefore isn't a default
                state_dict = self.ds_engine._zero3_consolidated_16bit_state_dict()
            else:
                raise NotImplementedError
        else:
            state_dict = self.ds_engine.module.state_dict()

        # only save ckpt in rank 0
        if deepspeed.comm.get_rank() == 0:
            ckpt = OrderedDict((k, state_dict[k]) for k in trainable_params)

            if epoch is None:
                torch.save(ckpt, os.path.join(save_dir, "ckpt.pt"))
            elif step is None:
                torch.save(ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch}.pt"))
            else:
                torch.save(
                    ckpt, os.path.join(save_dir, f"ckpt_epoch{epoch}_step{step}.pt")
                )
            # save tokenizer
            self.model.tokenizer.save_pretrained(save_dir)
            # save configuration
            self.model.llm.config.save_pretrained(save_dir)
            logging.info(f"[!] Save model in {save_dir}")
