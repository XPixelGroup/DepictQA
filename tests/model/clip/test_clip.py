import torch
from PIL import Image

from model.clip import load_clip

if __name__ == "__main__":
    img_path = "./tests/model/clip/a-motorcycle-381361.png"
    clip_path = "/root/.cache/clip/ViT-L-14.pt"
    training = True
    vision_preprocess = {
        "patch_size": 14,
        "resize": 224,
        "max_size": 672,
        "crop_ratio": [0.7, 1.0],
        "keep_ratio": True,
    }

    clip_encoder, visual_preprocess = load_clip(
        clip_path, training, vision_preprocess, device="cuda"
    )
    visual_encoder = clip_encoder.visual.cuda().half()

    # === Local & Global Features === #
    image = Image.open(img_path)
    image = visual_preprocess(image).to(torch.float16).cuda().unsqueeze(0)
    print("=" * 100)
    print("Image Shape: ", image.shape)
    with torch.no_grad():
        local_embeddings = visual_encoder.forward_patch_features(image)
        global_embeddings = visual_encoder(image)
    print("Local Embeddings: ", local_embeddings.shape)
    print("Global Embeddings: ", global_embeddings.shape)
