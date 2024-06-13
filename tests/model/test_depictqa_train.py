import torch
from easydict import EasyDict

from model.depictqa import DepictQA

if __name__ == "__main__":
    conv_type = "vicuna_v1"
    unique_tag = True
    no_ref = False
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

    img_embs = torch.rand((1, 25, 4096)).to(torch.float16).cuda()
    img_A_embs = torch.rand((1, 25, 4096)).to(torch.float16).cuda()
    img_B_embs = torch.rand((1, 25, 4096)).to(torch.float16).cuda()
    img_embs_list = [img_embs, img_A_embs, img_B_embs]
    model = DepictQA(args, training=True).cuda()

    roles = model.roles
    seps = model.seps
    print(f'[roles]: "{roles[0]}", "{roles[1]}"')
    print(f'[seps]: "{seps[0]}", "{seps[1]}"')
    seps = [sep[1:-1] for sep in seps]  # remove " " in the beginning and ending
    print(f'[original seps]: "{seps[0]}", "{seps[1]}"')

    prompt_base = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
    prompt_quality = (
        " As an AI assistant, you are performing an image quality assessment task."
    )
    prompt_ref = " A high-quality Reference Image is provided to assist the evaluation."

    ##############################
    # quality_compare
    ##############################

    task_type = "quality_compare_noref" if no_ref else "quality_compare"
    text = "What differences do you notice when comparing the overall quality of Image A and Image B, and can you explain these discrepancies?"
    answer = "Image A slightly surpasses Image B in terms of overall quality and texture quality. Although Image A is noticeably inferior to Image B in color distortion, it exhibits superiority in noise handling."
    conversations = [
        [{"from": "human", "value": text}, {"from": "gpt", "value": answer}]
    ]
    input_ids, tgt_ids, attn_mask = model.tokenize_conv(conversations, task_type)
    inputs_embs, tgts, attn_mask = model.fuse_vision_iqa(
        img_embs_list,
        input_ids,
        tgt_ids,
        attn_mask,
        task_type,
    )

    if no_ref:
        if unique_tag:
            text_part0 = f"{prompt_base}{prompt_quality}\n\n {seps[1]} Image A: <Img-A>"
            text_part1 = f"</Img-A>\n\n {seps[1]} Image B: <Img-B>"
            text_part2 = f"</Img-B>\n\n {seps[1]} {roles[0]}: What differences do you notice when comparing the overall quality of Image A and Image B, and can you explain these discrepancies?\n {seps[0]} {roles[1]}: "
            text_part3 = f"Image A slightly surpasses Image B in terms of overall quality and texture quality. Although Image A is noticeably inferior to Image B in color distortion, it exhibits superiority in noise handling.\n {seps[1]}"
        else:
            text_part0 = f"{prompt_base}{prompt_quality}\n\n {seps[1]} Image A: <Img>"
            text_part1 = f"</Img>\n\n {seps[1]} Image B: <Img>"
            text_part2 = f"</Img>\n\n {seps[1]} {roles[0]}: What differences do you notice when comparing the overall quality of Image A and Image B, and can you explain these discrepancies?\n {seps[0]} {roles[1]}: "
            text_part3 = f"Image A slightly surpasses Image B in terms of overall quality and texture quality. Although Image A is noticeably inferior to Image B in color distortion, it exhibits superiority in noise handling.\n {seps[1]}"

        token_part0 = model.tokenizer(
            text_part0, return_tensors="pt", add_special_tokens=False
        )
        token_part0_ids = token_part0.input_ids.expand(1, -1).cuda()
        token_part1 = model.tokenizer(
            text_part1, return_tensors="pt", add_special_tokens=False
        )
        token_part1_ids = token_part1.input_ids.expand(1, -1).cuda()
        token_part2 = model.tokenizer(
            text_part2, return_tensors="pt", add_special_tokens=False
        )
        token_part2_ids = token_part2.input_ids.expand(1, -1).cuda()
        token_part3 = model.tokenizer(
            text_part3, return_tensors="pt", add_special_tokens=False
        )
        token_part3_ids = token_part3.input_ids.expand(1, -1).cuda()

        embed_part0 = model.llm.model.model.embed_tokens(token_part0_ids)
        embed_part1 = model.llm.model.model.embed_tokens(token_part1_ids)
        embed_part2 = model.llm.model.model.embed_tokens(token_part2_ids)
        embed_part3 = model.llm.model.model.embed_tokens(token_part3_ids)

        bos = (
            torch.ones(
                [1, 1], dtype=token_part0_ids.dtype, device=token_part0_ids.device
            )
            * model.tokenizer.bos_token_id
        )
        bos_embs = model.llm.model.model.embed_tokens(bos)
        gt_embs = torch.cat(
            [
                bos_embs,
                embed_part0,
                img_A_embs,
                embed_part1,
                img_B_embs,
                embed_part2,
                embed_part3,
            ],
            dim=1,
        )
    else:
        if unique_tag:
            text_part0 = f"{prompt_base}{prompt_quality}{prompt_ref}\n\n {seps[1]} Reference Image: <Img-Reference>"
            text_part1 = f"</Img-Reference>\n\n {seps[1]} Image A: <Img-A>"
            text_part2 = f"</Img-A>\n\n {seps[1]} Image B: <Img-B>"
            text_part3 = f"</Img-B>\n\n {seps[1]} {roles[0]}: What differences do you notice when comparing the overall quality of Image A and Image B, and can you explain these discrepancies?\n {seps[0]} {roles[1]}: "
            text_part4 = f"Image A slightly surpasses Image B in terms of overall quality and texture quality. Although Image A is noticeably inferior to Image B in color distortion, it exhibits superiority in noise handling.\n {seps[1]}"
        else:
            text_part0 = f"{prompt_base}{prompt_quality}{prompt_ref}\n\n {seps[1]} Reference Image: <Img>"
            text_part1 = f"</Img>\n\n {seps[1]} Image A: <Img>"
            text_part2 = f"</Img>\n\n {seps[1]} Image B: <Img>"
            text_part3 = f"</Img>\n\n {seps[1]} {roles[0]}: What differences do you notice when comparing the overall quality of Image A and Image B, and can you explain these discrepancies?\n {seps[0]} {roles[1]}: "
            text_part4 = f"Image A slightly surpasses Image B in terms of overall quality and texture quality. Although Image A is noticeably inferior to Image B in color distortion, it exhibits superiority in noise handling.\n {seps[1]}"

        token_part0 = model.tokenizer(
            text_part0, return_tensors="pt", add_special_tokens=False
        )
        token_part0_ids = token_part0.input_ids.expand(1, -1).cuda()
        token_part1 = model.tokenizer(
            text_part1, return_tensors="pt", add_special_tokens=False
        )
        token_part1_ids = token_part1.input_ids.expand(1, -1).cuda()
        token_part2 = model.tokenizer(
            text_part2, return_tensors="pt", add_special_tokens=False
        )
        token_part2_ids = token_part2.input_ids.expand(1, -1).cuda()
        token_part3 = model.tokenizer(
            text_part3, return_tensors="pt", add_special_tokens=False
        )
        token_part3_ids = token_part3.input_ids.expand(1, -1).cuda()
        token_part4 = model.tokenizer(
            text_part4, return_tensors="pt", add_special_tokens=False
        )
        token_part4_ids = token_part4.input_ids.expand(1, -1).cuda()

        embed_part0 = model.llm.model.model.embed_tokens(token_part0_ids)
        embed_part1 = model.llm.model.model.embed_tokens(token_part1_ids)
        embed_part2 = model.llm.model.model.embed_tokens(token_part2_ids)
        embed_part3 = model.llm.model.model.embed_tokens(token_part3_ids)
        embed_part4 = model.llm.model.model.embed_tokens(token_part4_ids)

        bos = (
            torch.ones(
                [1, 1], dtype=token_part0_ids.dtype, device=token_part0_ids.device
            )
            * model.tokenizer.bos_token_id
        )
        bos_embs = model.llm.model.model.embed_tokens(bos)
        gt_embs = torch.cat(
            [
                bos_embs,
                embed_part0,
                img_embs,
                embed_part1,
                img_A_embs,
                embed_part2,
                img_B_embs,
                embed_part3,
                embed_part4,
            ],
            dim=1,
        )

    assert (inputs_embs == gt_embs).all()
    print("[Embedding right]: quality_compare")

    ##############################
    # quality_single
    ##############################

    task_type = "quality_single_B_noref" if no_ref else "quality_single_B"
    text = "Analyze the image's quality, focusing on its key elements, and detail your findings."
    answer = "This is the answer."
    conversations = [
        [{"from": "human", "value": text}, {"from": "gpt", "value": answer}]
    ]
    input_ids, tgt_ids, attn_mask = model.tokenize_conv(conversations, task_type)
    inputs_embs, tgts, attn_mask = model.fuse_vision_iqa(
        img_embs_list,
        input_ids,
        tgt_ids,
        attn_mask,
        task_type,
    )
    if no_ref:
        if unique_tag:
            text_part0 = f"{prompt_base}{prompt_quality}\n\n {seps[1]} Image: <Img>"
            text_part1 = f"</Img>\n\n {seps[1]} {roles[0]}: Analyze the image's quality, focusing on its key elements, and detail your findings.\n {seps[0]} {roles[1]}: "
            text_part2 = f"This is the answer.\n {seps[1]}"
        else:
            text_part0 = f"{prompt_base}{prompt_quality}\n\n {seps[1]} Image: <Img>"
            text_part1 = f"</Img>\n\n {seps[1]} {roles[0]}: Analyze the image's quality, focusing on its key elements, and detail your findings.\n {seps[0]} {roles[1]}: "
            text_part2 = f"This is the answer.\n {seps[1]}"

        token_part0 = model.tokenizer(
            text_part0, return_tensors="pt", add_special_tokens=False
        )
        token_part0_ids = token_part0.input_ids.expand(1, -1).cuda()
        token_part1 = model.tokenizer(
            text_part1, return_tensors="pt", add_special_tokens=False
        )
        token_part1_ids = token_part1.input_ids.expand(1, -1).cuda()
        token_part2 = model.tokenizer(
            text_part2, return_tensors="pt", add_special_tokens=False
        )
        token_part2_ids = token_part2.input_ids.expand(1, -1).cuda()

        embed_part0 = model.llm.model.model.embed_tokens(token_part0_ids)
        embed_part1 = model.llm.model.model.embed_tokens(token_part1_ids)
        embed_part2 = model.llm.model.model.embed_tokens(token_part2_ids)

        bos = (
            torch.ones(
                [1, 1], dtype=token_part0_ids.dtype, device=token_part0_ids.device
            )
            * model.tokenizer.bos_token_id
        )
        bos_embs = model.llm.model.model.embed_tokens(bos)

        if "quality_single_A" in task_type:
            gt_embs = torch.cat(
                [bos_embs, embed_part0, img_A_embs, embed_part1, embed_part2], dim=1
            )
        elif "quality_single_B" in task_type:
            gt_embs = torch.cat(
                [bos_embs, embed_part0, img_B_embs, embed_part1, embed_part2], dim=1
            )
    else:
        if unique_tag:
            text_part0 = (
                f"{prompt_base}{prompt_quality}{prompt_ref}\n\n {seps[1]} Reference Image: <Img-Reference>",
            )
            text_part1 = f"</Img-Reference>\n\n {seps[1]} Image: <Img>"
            text_part2 = f"</Img>\n\n {seps[1]} {roles[0]}: Analyze the image's quality, focusing on its key elements, and detail your findings.\n {seps[0]} {roles[1]}: "
            text_part3 = f"This is the answer.\n {seps[1]}"
        else:
            text_part0 = (
                f"{prompt_base}{prompt_quality}{prompt_ref}\n\n {seps[1]} Reference Image: <Img>",
            )
            text_part1 = f"</Img>\n\n {seps[1]} Image: <Img>"
            text_part2 = f"</Img>\n\n {seps[1]} {roles[0]}: Analyze the image's quality, focusing on its key elements, and detail your findings.\n {seps[0]} {roles[1]}: "
            text_part3 = f"This is the answer.\n {seps[1]}"

        token_part0 = model.tokenizer(
            text_part0, return_tensors="pt", add_special_tokens=False
        )
        token_part0_ids = token_part0.input_ids.expand(1, -1).cuda()
        token_part1 = model.tokenizer(
            text_part1, return_tensors="pt", add_special_tokens=False
        )
        token_part1_ids = token_part1.input_ids.expand(1, -1).cuda()
        token_part2 = model.tokenizer(
            text_part2, return_tensors="pt", add_special_tokens=False
        )
        token_part2_ids = token_part2.input_ids.expand(1, -1).cuda()
        token_part3 = model.tokenizer(
            text_part3, return_tensors="pt", add_special_tokens=False
        )
        token_part3_ids = token_part3.input_ids.expand(1, -1).cuda()

        embed_part0 = model.llm.model.model.embed_tokens(token_part0_ids)
        embed_part1 = model.llm.model.model.embed_tokens(token_part1_ids)
        embed_part2 = model.llm.model.model.embed_tokens(token_part2_ids)
        embed_part3 = model.llm.model.model.embed_tokens(token_part3_ids)

        bos = (
            torch.ones(
                [1, 1], dtype=token_part0_ids.dtype, device=token_part0_ids.device
            )
            * model.tokenizer.bos_token_id
        )
        bos_embs = model.llm.model.model.embed_tokens(bos)

        if "quality_single_A" in task_type:
            gt_embs = torch.cat(
                [
                    bos_embs,
                    embed_part0,
                    img_embs,
                    embed_part1,
                    img_A_embs,
                    embed_part2,
                    embed_part3,
                ],
                dim=1,
            )
        elif "quality_single_B" in task_type:
            gt_embs = torch.cat(
                [
                    bos_embs,
                    embed_part0,
                    img_embs,
                    embed_part1,
                    img_B_embs,
                    embed_part2,
                    embed_part3,
                ],
                dim=1,
            )

    assert (inputs_embs == gt_embs).all()
    print("[Embedding right]: quality_single")
