import argparse
import asyncio
import json
import os
import pprint
import threading
import time
import uuid
from functools import partial
from threading import Thread

import requests
import torch
import uvicorn
import yaml
from easydict import EasyDict
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import TextIteratorStreamer

from model.depictqa import DepictQA

from .utils import (
    IMAGE_TOKEN_INDEX,
    WORKER_HEART_BEAT_INTERVAL,
    build_logger,
    pretty_print_semaphore,
    server_error_msg,
    tokenizer_image_token,
)

worker_id = str(uuid.uuid4())[:6]
global_counter = 0
model_semaphore = None


def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


class DepictQAWorker:
    def __init__(self, cfg):
        assert os.path.exists(
            cfg.model["vision_encoder_path"]
        ), "vision_encoder_path not exist!"
        assert os.path.exists(cfg.model["llm_path"]), "llm_path not exist!"
        assert os.path.exists(cfg.model["delta_path"]), "delta_path not exist!"

        self.model_name = cfg.serve["model_name"]
        self.controller = cfg.serve["controller"]
        self.worker_url = cfg.serve["worker_url"]
        self.worker_id = worker_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.model = DepictQA(cfg, training=False).eval().half().cuda()

        delta_path = cfg.model["delta_path"]
        delta_ckpt = torch.load(delta_path, map_location=torch.device("cpu"))
        self.model.load_state_dict(delta_ckpt, strict=False)
        logger.info(f"[!] Load pretrained delta ckpt from {delta_path}")
        logger.info(f"[!] init DepictQA over ...")

        self.is_multimodal = True

        if not cfg.serve.no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,)
            )
            self.heart_beat_thread.start()
        self.img_paths = []

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller + "/register_worker"
        data = {
            "worker_name": self.worker_url,
            "check_heart_beat": True,
            "worker_status": self.get_status(),
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(
            f"Send heart beat. Models: {[self.model_name]}. "
            f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
            f"global_counter: {global_counter}"
        )

        url = self.controller + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(
                    url,
                    json={
                        "worker_name": self.worker_url,
                        "queue_length": self.get_queue_length(),
                    },
                    timeout=5,
                )
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return (
                cfg.serve.limit_model_concurrency
                - model_semaphore._value
                + (
                    len(model_semaphore._waiters)
                    if model_semaphore._waiters is not None
                    else 0
                )
            )

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.model.tokenizer, self.model
        query = params["query"]
        task_type = params["task_type"]
        logger.info(f"Task: {task_type}")

        img_paths = params.get("img_paths", None)
        if img_paths is None:
            logger.warning("No img_paths received")
        img_path = img_paths[0]
        img_A_path = img_paths[1]
        img_B_path = img_paths[2]
        logger.info(f"Reference path: {img_path}")
        logger.info(f"Image A path: {img_A_path}")
        logger.info(f"Image B path: {img_B_path}")

        temperature = float(params.get("temperature", 0.0))
        top_p = float(params.get("top_p", 0.9))
        max_all_tokens = params.get("max_all_tokens", 512)
        max_new_tokens = min(int(params.get("max_new_tokens", 400)), 1024)
        stop_str = params.get("stop", None)

        input_ids = (
            tokenizer_image_token(
                query, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )
        streamer = TextIteratorStreamer(
            tokenizer, skip_query=True, skip_special_tokens=True, timeout=50
        )

        max_new_tokens = min(max_new_tokens, max_all_tokens - input_ids.shape[-1])
        if max_new_tokens < 1:
            yield json.dumps(
                {
                    "text": query
                    + "Exceeds max token length. Please start a new conversation, thanks.",
                    "error_code": 0,
                }
            ).encode() + b"\0"
            return
        logger.info(f"query: {query}")
        inputs = dict(
            query=[query],
            img_path=[img_path],
            img_A_path=[img_A_path],
            img_B_path=[img_B_path],
            top_p=top_p,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            task_type=task_type,
            streamer=streamer,
        )
        thread = Thread(target=model.generate, kwargs={"inputs": inputs})
        thread.start()

        generated_text = query
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[: -len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            logger.info("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            logger.info("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            logger.info("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(cfg.serve.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(
        partial(release_model_semaphore, fn=worker.send_heart_beat)
    )
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="config.yaml")
    parser.add_argument("--cfg_serve", type=str, default="serve.yaml")
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    with open(args.cfg_serve, "r") as f:
        cfg_serve = EasyDict(yaml.safe_load(f))
    cfg["serve"] = cfg_serve["worker"]

    os.makedirs(cfg_serve["log_dir"], exist_ok=True)
    logger = build_logger(
        "model_worker",
        os.path.join(cfg_serve["log_dir"], f"model_worker_{worker_id}.log"),
    )
    logger.info("cfg_serve: {}".format(pprint.pformat(cfg_serve)))
    logger.info("cfg_model: {}".format(pprint.pformat(cfg)))

    worker = DepictQAWorker(cfg)
    uvicorn.run(
        app, host=cfg_serve.worker.host, port=cfg_serve.worker.port, log_level="info"
    )
