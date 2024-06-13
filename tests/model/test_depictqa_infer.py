import torch
from easydict import EasyDict

from model.depictqa import DepictQA

if __name__ == "__main__":
    conv_type = "vicuna_v1"
    unique_tag = True
    no_ref = False
    img_path = "/opt/data/private/142/Datasets/BAPPS/twoafc_val_s64/val/traditional/ref/000000.png"
    img_A_path = "/opt/data/private/142/Datasets/BAPPS/twoafc_val_s64/val/traditional/p0/000000.png"
    img_B_path = "/opt/data/private/142/Datasets/BAPPS/twoafc_val_s64/val/traditional/p1/000000.png"
    args = {
        "model": {
            "vision_encoder_path": "/root/.cache/clip/ViT-L-14.pt",
            "vision_feature_type": "local",
            "llm_path": "/opt/data/private/142/Model_zoo/LLM/vicuna/vicuna-7b-v1.5/",
            "llm_conv_type": conv_type,
            "lora": {
                "lora_r": 16,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "unique_tag": unique_tag,
            "max_tgt_len": 400,
        },
        "vision_preprocess": {
            "patch_size": 14,
            "resize": 224,
            "max_size": 672,
            "crop_ratio": [0.7, 1.0],
            "keep_ratio": True,
        },
    }
    args = EasyDict(args)

    model = DepictQA(args, training=False).eval().half().cuda()

    roles = model.roles
    seps = model.seps
    print(f'[roles]: "{roles[0]}", "{roles[1]}"')
    print(f'[seps]: "{seps[0]}", "{seps[1]}"')
    seps = [sep[1:-1] for sep in seps]  # remove " " in the beginning and ending
    print(f'[original seps]: "{seps[0]}", "{seps[1]}"')

    def get_embs_from_text(text):
        tokens = model.tokenizer(
            text, return_tensors="pt", add_special_tokens=False
        ).input_ids.cuda()
        embs = model.llm.model.model.embed_tokens(tokens).expand(1, -1, -1).cuda()
        return embs

    prompt_base = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    prompt_quality = (
        " As an AI assistant, you are performing an image quality assessment task."
    )
    prompt_ref = " A high-quality Reference Image is provided to assist the evaluation."

    ##############################
    # quality_compare
    ##############################

    inputs = {
        "query": ["You are very good."],
        "task_type": "quality_compare_noref" if no_ref else "quality_compare",
        "img_path": [img_path],
        "img_A_path": [img_A_path],
        "img_B_path": [img_B_path],
        "max_tgt_len": 400,
    }
    input_embs, _ = model.get_generate_embs(inputs)

    img_embs = model.emb_img(inputs["img_path"])
    img_A_embs = model.emb_img(inputs["img_A_path"])
    img_B_embs = model.emb_img(inputs["img_B_path"])

    if no_ref:
        if unique_tag:
            text1 = f"{prompt_base}{prompt_quality}\n\n {seps[1]} Image A: <Img-A>"
            text2 = f"</Img-A>\n\n {seps[1]} Image B: <Img-B>"
            text3 = f"</Img-B>\n\n {seps[1]} {roles[0]}: You are very good.\n {seps[0]} {roles[1]}: "
        else:
            text1 = f"{prompt_base}{prompt_quality}\n\n {seps[1]} Image A: <Img>"
            text2 = f"</Img>\n\n {seps[1]} Image B: <Img>"
            text3 = f"</Img>\n\n {seps[1]} {roles[0]}: You are very good.\n {seps[0]} {roles[1]}: "

        embs1 = get_embs_from_text(text1)
        embs2 = get_embs_from_text(text2)
        embs3 = get_embs_from_text(text3)

        bos = torch.ones([1, 1], dtype=torch.long).cuda() * model.tokenizer.bos_token_id
        bos_embs = model.llm.model.model.embed_tokens(bos)
        gt_embs = torch.cat(
            [bos_embs, embs1, img_A_embs, embs2, img_B_embs, embs3],
            dim=1,
        )
    else:
        if unique_tag:
            text1 = f"{prompt_base}{prompt_quality}{prompt_ref}\n\n {seps[1]} Reference Image: <Img-Reference>"
            text2 = f"</Img-Reference>\n\n {seps[1]} Image A: <Img-A>"
            text3 = f"</Img-A>\n\n {seps[1]} Image B: <Img-B>"
            text4 = f"</Img-B>\n\n {seps[1]} {roles[0]}: You are very good.\n {seps[0]} {roles[1]}: "
        else:
            text1 = f"{prompt_base}{prompt_quality}{prompt_ref}\n\n {seps[1]} Reference Image: <Img>"
            text2 = f"</Img>\n\n {seps[1]} Image A: <Img>"
            text3 = f"</Img>\n\n {seps[1]} Image B: <Img>"
            text4 = f"</Img>\n\n {seps[1]} {roles[0]}: You are very good.\n {seps[0]} {roles[1]}: "

        embs1 = get_embs_from_text(text1)
        embs2 = get_embs_from_text(text2)
        embs3 = get_embs_from_text(text3)
        embs4 = get_embs_from_text(text4)

        bos = torch.ones([1, 1], dtype=torch.long).cuda() * model.tokenizer.bos_token_id
        bos_embs = model.llm.model.model.embed_tokens(bos)
        gt_embs = torch.cat(
            [
                bos_embs,
                embs1,
                img_embs,
                embs2,
                img_A_embs,
                embs3,
                img_B_embs,
                embs4,
            ],
            dim=1,
        )

    assert (input_embs == gt_embs).all()
    print("[Embedding right]: quality_compare")

    ##############################
    # quality_single
    ##############################

    inputs = {
        "query": ["You are very good."],
        "task_type": "quality_single_A_noref" if no_ref else "quality_single_A",
        "img_path": [img_path],
        "img_A_path": [img_A_path],
        "img_B_path": [img_B_path],
        "max_tgt_len": 400,
    }
    input_embs, _ = model.get_generate_embs(inputs)

    img_embs = model.emb_img(inputs["img_path"])
    img_A_embs = model.emb_img(inputs["img_A_path"])

    if no_ref:
        if unique_tag:
            text1 = f"{prompt_base}{prompt_quality}\n\n {seps[1]} Image: <Img>"
            text2 = f"</Img>\n\n {seps[1]} {roles[0]}: You are very good.\n {seps[0]} {roles[1]}: "
        else:
            text1 = f"{prompt_base}{prompt_quality}\n\n {seps[1]} Image: <Img>"
            text2 = f"</Img>\n\n {seps[1]} {roles[0]}: You are very good.\n {seps[0]} {roles[1]}: "

        embs1 = get_embs_from_text(text1)
        embs2 = get_embs_from_text(text2)

        bos = torch.ones([1, 1], dtype=torch.long).cuda() * model.tokenizer.bos_token_id
        bos_embs = model.llm.model.model.embed_tokens(bos)
        gt_embs = torch.cat([bos_embs, embs1, img_A_embs, embs2], dim=1)
    else:
        if unique_tag:
            text1 = f"{prompt_base}{prompt_quality}{prompt_ref}\n\n {seps[1]} Reference Image: <Img-Reference>"
            text2 = f"</Img-Reference>\n\n {seps[1]} Image: <Img>"
            text3 = f"</Img>\n\n {seps[1]} {roles[0]}: You are very good.\n {seps[0]} {roles[1]}: "
        else:
            text1 = f"{prompt_base}{prompt_quality}{prompt_ref}\n\n {seps[1]} Reference Image: <Img>"
            text2 = f"</Img>\n\n {seps[1]} Image: <Img>"
            text3 = f"</Img>\n\n {seps[1]} {roles[0]}: You are very good.\n {seps[0]} {roles[1]}: "

        embs1 = get_embs_from_text(text1)
        embs2 = get_embs_from_text(text2)
        embs3 = get_embs_from_text(text3)

        bos = torch.ones([1, 1], dtype=torch.long).cuda() * model.tokenizer.bos_token_id
        bos_embs = model.llm.model.model.embed_tokens(bos)
        gt_embs = torch.cat(
            [bos_embs, embs1, img_embs, embs2, img_A_embs, embs3], dim=1
        )

    assert (input_embs == gt_embs).all()
    print("[Embedding right]: quality_single")
