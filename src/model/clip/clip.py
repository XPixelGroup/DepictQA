import hashlib
import math
import os
import random
import urllib
import warnings
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from pkg_resources import packaging
from torchvision.transforms.functional import center_crop
from tqdm import tqdm

from .model_clip import build_model

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
    NEAREST = InterpolationMode.NEAREST
except ImportError:
    BICUBIC = Image.BICUBIC
    NEAREST = Image.NEAREST


if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")


__all__ = ["available_models", "load_clip"]

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if (
            hashlib.sha256(open(download_target, "rb").read()).hexdigest()
            == expected_sha256
        ):
            return download_target
        else:
            warnings.warn(
                f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
            )

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target


class CustomTransform:
    def __init__(
        self,
        patch_size=14,
        resize=224,
        min_size=70,
        max_size=672,
        crop_ratio=[0.7, 1.0],
        keep_ratio=True,
        training=True,
    ):
        self.patch_size = patch_size
        self.resize = resize
        self.min_size = min_size
        self.max_size = max_size
        self.crop_ratio = crop_ratio
        self.keep_ratio = keep_ratio
        self.random_crop = True if training else False
        self.mean_cuda = torch.tensor((0.48145466, 0.4578275, 0.40821073))[
            :, None, None
        ].cuda()
        self.std_cuda = torch.tensor((0.26862954, 0.26130258, 0.27577711))[
            :, None, None
        ].cuda()

    def __call__(self, img):
        h, w = img.height, img.width
        if self.resize <= min(h, w) or self.max_size <= max(h, w):
            # resize
            if self.keep_ratio:
                ratio = min(self.resize / min(h, w), self.max_size / max(h, w))
                h_new, w_new = round(h * ratio), round(w * ratio)
                img = img.resize((w_new, h_new), resample=Image.BICUBIC)
            else:
                img = img.resize((self.resize, self.resize), resample=Image.BICUBIC)
        # random crop
        h, w = img.height, img.width
        if self.random_crop and self.min_size < min(h, w):
            crop_ratio_min = self.min_size / min(h, w)
            crop_ratio_h = max(random.uniform(self.crop_ratio[0], self.crop_ratio[1]), crop_ratio_min)
            crop_ratio_w = max(random.uniform(self.crop_ratio[0], self.crop_ratio[1]), crop_ratio_min)
            h_crop, w_crop = round(h * crop_ratio_h), round(w * crop_ratio_w)
            img = center_crop(img, (h_crop, w_crop))
        # padding to 14 X N shapes
        h, w = max(self.min_size, img.height), max(self.min_size, img.width)
        h_pad, w_pad = [
            int(math.ceil(_ / self.patch_size) * self.patch_size) for _ in [h, w]
        ]
        img = center_crop(img, (h_pad, w_pad))
        # to tensor & normalize
        """
        # For AMD cpu, handling tensor is quite slow (10x-15x more time), thus 
        # 1. do not use ToTensor & Normalize in torchvision.transforms. 
        # 2. torch.tensor().cuda() to replace ToTensor. 
        # 3. (img.cuda() - mean.cuda()) / std.cuda() to replace Normalize. 
        """
        img = np.array(img) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).cuda()
        img = (img - self.mean_cuda) / self.std_cuda
        return img


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_clip(
    name: str,
    training: bool,
    vision_preprocess: dict,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit: bool = False,
    download_root: str = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(
            _MODELS[name], download_root or os.path.expanduser("~/.cache/clip")
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    with open(model_path, "rb") as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(
                opened_file, map_location=device if jit else "cpu"
            ).eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(
                    f"File {model_path} is not a JIT archive. Loading as a state dict instead"
                )
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, CustomTransform(**vision_preprocess, training=training)

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, CustomTransform(**vision_preprocess, training=training)
