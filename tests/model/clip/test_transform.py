import numpy as np
from PIL import Image

from model.clip.clip import CustomTransform

if __name__ == "__main__":
    img_path = "./tests/model/clip/a-motorcycle-381361.png"
    patch_size = 14
    resize = 224
    max_size = 672
    crop_ratio = [0.7, 1.0]
    keep_ratio = True
    training = True
    fn_transform = CustomTransform(
        patch_size, resize, max_size, crop_ratio, keep_ratio, training
    )
    img = Image.open(img_path)
    print(f"Original shape: {img.height}, {img.width}")
    img = fn_transform(img)
    img = img.permute(1, 2, 0).cpu().numpy()

    mean = np.array((0.48145466, 0.4578275, 0.40821073))[None, None, ...]
    std = np.array((0.26862954, 0.26130258, 0.27577711))[None, None, ...]
    img = np.uint8((img * std + mean) * 255)

    print(f"Transformed shape: {img.shape}")
    img = Image.fromarray(img)
    img.save("./tests/model/clip/a-motorcycle-381361-transform.png")
