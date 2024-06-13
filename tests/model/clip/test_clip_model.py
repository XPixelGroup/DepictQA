import torch

from model.clip.model_clip import build_model

if __name__ == "__main__":
    model_path = "/root/.cache/clip/ViT-L-14.pt"
    with open(model_path, "rb") as opened_file:
        model = torch.jit.load(opened_file, map_location="cpu").eval()
    state_dict = model.state_dict()
    model = build_model(state_dict)
    print("=" * 100)
    print("Full Image Resolution: ", model.visual.input_resolution)
    print("Full Position Embedding: ", state_dict["visual.positional_embedding"].shape)

    # === Local & Global Features === #
    visual_encoder = model.visual.cuda().half()
    # 150 tokens
    image = torch.rand(2, 3, 14 * 15, 14 * 10).to(torch.float16).cuda()
    with torch.no_grad():
        local_embeddings = visual_encoder.forward_patch_features(image)
        global_embeddings = visual_encoder(image)
    print("=" * 100)
    print("Local Embeddings: ", local_embeddings.shape)
    print("Global Embeddings: ", global_embeddings.shape)

    # 125 tokens
    image = torch.rand(2, 3, 14 * 25, 14 * 5).to(torch.float16).cuda()
    with torch.no_grad():
        local_embeddings = visual_encoder.forward_patch_features(image)
        global_embeddings = visual_encoder(image)
    print("=" * 100)
    print("Local Embeddings: ", local_embeddings.shape)
    print("Global Embeddings: ", global_embeddings.shape)
