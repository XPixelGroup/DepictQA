import argparse
import datetime
import hashlib
import json
import os
import pprint
import time

import gradio as gr
import requests
import yaml
from easydict import EasyDict

from model.conversations import SeparatorStyle, conversation_dict, default_conversation

from .utils import build_logger, moderation_msg, server_error_msg, violates_moderation

headers = {"User-Agent": "DepictQA Client"}

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def get_task(img_ref, img_A, img_B):
    prefix = "quality_"
    suffix = ""
    if img_ref is None:
        suffix = "_noref"
    if img_A is not None and img_B is not None:
        task_name = "compare"
    elif img_A is not None and img_B is None:
        task_name = "single_A"
    elif img_A is None and img_B is not None:
        task_name = "single_B"
    else:
        raise ValueError("Either Image A or Image B should be given.")
    return prefix + task_name + suffix


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(cfg["log_dir"], f"{t.year}-{t.month:02d}-{t.day:02d}-conv.json")
    return name


def get_model_list():
    ret = requests.post(cfg["controller_url"] + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(cfg["controller_url"] + "/list_models")
    models = ret.json()["models"]
    logger.info(f"Models: {models}")
    return models


get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def load_demo(url_params, request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}. params: {url_params}")

    dropdown_update = gr.Dropdown.update(visible=True)
    if "model" in url_params:
        model = url_params["model"]
        if model in models:
            dropdown_update = gr.Dropdown.update(value=model, visible=True)

    state = default_conversation.copy()
    return state, dropdown_update


def load_demo_refresh_model_list(request: gr.Request):
    logger.info(f"load_demo. ip: {request.client.host}")
    models = get_model_list()
    state = default_conversation.copy()
    dropdown_update = gr.Dropdown.update(
        choices=models, value=models[0] if len(models) > 0 else ""
    )
    return state, dropdown_update


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(time.time(), 4),
            "type": vote_type,
            "model": model_selector,
            "state": state.dict(),
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


def upvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def flag_last_response(state, model_selector, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", model_selector, request)
    return ("",) + (disable_btn,) * 3


def regenerate(state, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    state.messages[-1][-1] = None
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = prev_human_msg[1][:2]
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None, None, None) + (disable_btn,) * 5


def add_text(state, text, img_ref, img_A, img_B, request: gr.Request):
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    input_images = [img_ref, img_A, img_B]
    images = []
    for image in input_images:
        images.append(image)
    if len(text) <= 0 and images is None:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), "") + (no_change_btn,) * 5
    if cfg["moderate"]:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            return (state, state.to_gradio_chatbot(), moderation_msg) + (
                no_change_btn,
            ) * 5

    text = text[:1536]  # Hard cut-off
    if images is not None:
        text = text[:1200]  # Hard cut-off for images
        text = (text, images)
        if len(state.get_images(return_pil=True)) > 0:
            state = default_conversation.copy()
    logger.info(f"{text}")
    state.append_message(state.roles[0], text)
    state.append_message(state.roles[1], None)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), "") + (disable_btn,) * 5


def http_bot(
    state,
    model_selector,
    temperature,
    top_p,
    max_new_tokens,
    max_all_tokens,
    request: gr.Request,
):
    logger.info(f"http_bot. ip: {request.client.host}")
    start_tstamp = time.time()
    model_name = model_selector
    if state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (state, state.to_gradio_chatbot()) + (no_change_btn,) * 5
        return

    if len(state.messages) == state.offset + 2:
        # First round of conversation
        assert "depictqa" in model_name.lower()
        if "vicuna_v0" in model_name.lower():
            template_name = "vicuna_v0"
        elif "vicuna_v1" in model_name.lower():
            template_name = "vicuna_v1"
        elif "llama_2" in model_name.lower():
            template_name = "llama_2"
        else:
            raise NotImplementedError(f"{model_name.lower()} is not supported")
        new_state = conversation_dict[template_name].copy()
        new_state.append_message(new_state.roles[0], state.messages[-2][1])
        new_state.append_message(new_state.roles[1], None)
        state = new_state

    # Query worker address
    controller_url = cfg["controller_url"]
    ret = requests.post(
        controller_url + "/get_worker_address", json={"model": model_name}
    )
    worker_addr = ret.json()["address"]
    logger.info(f"model_name: {model_name}, worker_addr: {worker_addr}")

    # No available worker
    if worker_addr == "":
        state.messages[-1][-1] = server_error_msg
        yield (
            state,
            state.to_gradio_chatbot(),
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    # Construct query
    query = state.get_query()
    imgs = state.get_images(return_pil=True)
    task_type = get_task(*imgs)

    img_hashs = []
    for img in imgs:
        img_hash = hashlib.md5(img.tobytes()).hexdigest() if img is not None else None
        img_hashs.append(img_hash)

    img_paths = []
    for img, img_hash in zip(imgs, img_hashs):
        if img is None:
            img_path = None
        else:
            t = datetime.datetime.now()
            img_path = os.path.join(
                cfg["log_dir"],
                "serve_images",
                f"{t.year}-{t.month:02d}-{t.day:02d}",
                f"{img_hash}.png",
            )
        img_paths.append(img_path)
        if img_path is not None and not os.path.isfile(img_path):
            os.makedirs(os.path.dirname(img_path), exist_ok=True)
            img.save(img_path)

    # Make requests
    pload = {
        "img_paths": img_paths,
        "query": query,
        "task_type": task_type,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_new_tokens": int(max_new_tokens),
        "max_all_tokens": int(max_all_tokens),
        "stop": (
            state.sep
            if state.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.MPT]
            else state.sep2
        ),
    }
    logger.info(f"==== request ====\n{pload}")

    state.messages[-1][-1] = "▌"
    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
    try:
        # Stream output
        response = requests.post(
            worker_addr + "/worker_generate_stream",
            headers=headers,
            json=pload,
            stream=True,
            timeout=10,
        )
        for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
            if chunk:
                data = json.loads(chunk.decode())
                if data["error_code"] == 0:
                    output = data["text"][len(query) :].strip()
                    state.messages[-1][-1] = output + "▌"
                    yield (state, state.to_gradio_chatbot()) + (disable_btn,) * 5
                else:
                    output = data["text"] + f" (error_code: {data['error_code']})"
                    state.messages[-1][-1] = output
                    yield (state, state.to_gradio_chatbot()) + (
                        disable_btn,
                        disable_btn,
                        disable_btn,
                        enable_btn,
                        enable_btn,
                    )
                    return
                time.sleep(0.03)
    except requests.exceptions.RequestException as e:
        state.messages[-1][-1] = server_error_msg
        yield (state, state.to_gradio_chatbot()) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.messages[-1][-1] = state.messages[-1][-1][:-1]
    yield (state, state.to_gradio_chatbot()) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{output}")

    with open(get_conv_log_filename(), "a") as fout:
        data = {
            "tstamp": round(finish_tstamp, 4),
            "type": "chat",
            "model": model_name,
            "start": round(start_tstamp, 4),
            "finish": round(finish_tstamp, 4),
            "state": state.dict(),
            "images": img_hashs,
            "ip": request.client.host,
        }
        fout.write(json.dumps(data) + "\n")


title_markdown = """
# 🌋 DepictQA: Depicted Image Quality Assessment with Vision Language Models
[[Project Page](https://depictqa.github.io/)] [[Code](https://github.com/XPixelGroup/DepictQA)] [[Model](https://github.com/XPixelGroup/DepictQA)]]
"""

distortion_markdown = """
### Distortion Identification
- __Inputs__: Image A / B, Reference Image (*Optional*)
- __Question Examples__: 
    - What are the most critical image quality issues in the evaluated image?
    - Identify the chief degradations in the evaluated image.
"""

assess_markdown = """
### Assessment Reasoning 
- __Inputs__: Image A / B, Reference Image (*Optional*)
- __Question Examples__: 
    - Please evaluate the image's quality and provide your reasons. 
    - What is your opinion on the quality of this image? Explain your viewpoint.
"""

rate_markdown = """
### Instant Rating
- __Inputs__: Image A & B, Reference Image (*Optional*)
- __Question Examples__: 
    - Determine which image exhibits higher quality between Image A and Image B.
    - Which of the two images, Image A or Image B, do you consider to be of better quality?
"""

compare_markdown = """
### Comparison Reasoning
- __Inputs__: Image A & B, Reference Image (*Optional*)
- __Question Examples__: 
    - Compare the overall quality of Image A with Image B and provide a comprehensive explanation.
    - Can you evaluate the overall quality in both Image A and Image B, and explain which one is superior?
"""

block_css = """

#buttons button {
    min-width: min(120px,100%);
}

"""


def build_demo(embed_mode):
    textbox = gr.Textbox(
        show_label=False, placeholder="Enter text and press ENTER", container=False
    )
    with gr.Blocks(title="DepictQA", theme=gr.themes.Default(), css=block_css) as demo:
        state = gr.State()

        if not embed_mode:
            gr.Markdown(title_markdown)

        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row(elem_id="model_selector_row"):
                    model_selector = gr.Dropdown(
                        choices=models,
                        value=models[0] if len(models) > 0 else "",
                        interactive=True,
                        show_label=False,
                        container=False,
                    )
                img_box = gr.Image(
                    type="pil", elem_id="img_box", label="Reference Image"
                )
                imgA_box = gr.Image(type="pil", elem_id="imgA_box", label="Image A")
                imgB_box = gr.Image(type="pil", elem_id="imgB_box", label="Image B")

                # cur_dir = os.path.dirname(os.path.abspath(__file__))
                # gr.Examples(examples=[
                #     [f"{cur_dir}/examples/extreme_ironing.jpg", "What is unusual about this image?"],
                #     [f"{cur_dir}/examples/waterview.jpg", "What are the things I should be cautious about when I visit here?"],
                # ], inputs=[img_box, textbox])

                with gr.Accordion("Parameters", open=False) as parameter_row:
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    max_output_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=400,
                        step=64,
                        interactive=True,
                        label="Max output tokens",
                    )
                    max_all_tokens = gr.Slider(
                        minimum=0,
                        maximum=1024,
                        value=512,
                        step=64,
                        interactive=True,
                        label="Max all tokens",
                    )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot", label="DepictQA Chatbot", height=550
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    regenerate_btn = gr.Button(
                        value="🔄  Regenerate", interactive=False
                    )
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        if not embed_mode:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(distortion_markdown)
                with gr.Column(scale=1):
                    gr.Markdown(assess_markdown)
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(rate_markdown)
                with gr.Column(scale=1):
                    gr.Markdown(compare_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False,
        )
        downvote_btn.click(
            downvote_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False,
        )
        flag_btn.click(
            flag_last_response,
            [state, model_selector],
            [textbox, upvote_btn, downvote_btn, flag_btn],
            queue=False,
        )

        regenerate_btn.click(
            regenerate, [state], [state, chatbot, textbox] + btn_list, queue=False
        ).then(
            http_bot,
            [
                state,
                model_selector,
                temperature,
                top_p,
                max_output_tokens,
                max_all_tokens,
            ],
            [state, chatbot] + btn_list,
        )

        clear_btn.click(
            clear_history,
            None,
            [state, chatbot, textbox, img_box, imgA_box, imgB_box] + btn_list,
            queue=False,
        )

        textbox.submit(
            add_text,
            [state, textbox, img_box, imgA_box, imgB_box],
            [state, chatbot, textbox] + btn_list,
            queue=False,
        ).then(
            http_bot,
            [
                state,
                model_selector,
                temperature,
                top_p,
                max_output_tokens,
                max_all_tokens,
            ],
            [state, chatbot] + btn_list,
        )

        submit_btn.click(
            add_text,
            [state, textbox, img_box, imgA_box, imgB_box],
            [state, chatbot, textbox] + btn_list,
            queue=False,
        ).then(
            http_bot,
            [
                state,
                model_selector,
                temperature,
                top_p,
                max_output_tokens,
                max_all_tokens,
            ],
            [state, chatbot] + btn_list,
        )

        if cfg["model_list_mode"] == "once":
            demo.load(
                load_demo,
                [url_params],
                [state, model_selector],
                _js=get_window_url_params,
                queue=False,
            )
        elif cfg["model_list_mode"] == "reload":
            demo.load(
                load_demo_refresh_model_list, None, [state, model_selector], queue=False
            )
        else:
            raise ValueError(f"Unknown model list mode: {cfg.model_list_mode}")

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="serve.yaml")
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.gradio["log_dir"] = cfg["log_dir"]
    cfg = cfg["gradio"]

    os.makedirs(cfg["log_dir"], exist_ok=True)
    logger = build_logger(
        "gradio_web_server", os.path.join(cfg["log_dir"], "gradio_web_server.log")
    )
    logger.info("cfg: {}".format(pprint.pformat(cfg)))

    models = get_model_list()
    demo = build_demo(cfg["embed"])
    demo.queue(concurrency_count=cfg["concurrency_count"], api_open=False).launch(
        server_name=cfg["host"], server_port=cfg["port"], share=cfg["share"]
    )
