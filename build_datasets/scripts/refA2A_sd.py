import argparse
import json
import os

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--read_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--exclude", action="store_true")
parser.add_argument("--training", action="store_true")


def check_include(degradation, severity, threshold):
    # For these distortions, only severity >= thre could be kept.
    distortions_thre = [
        "brighten",
        "darken",
        "contrast_weaken",
        "contrast_strengthen",
        "saturate_weaken",
        "saturate_strengthen",
        "quantization",
        "oversharpen",
    ]
    if not degradation:
        assert severity == 0
        return True
    for distortion in distortions_thre:
        if distortion in degradation and severity < threshold:
            return False
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    read_path = args.read_path
    save_path = args.save_path
    exclude = args.exclude
    training = args.training

    threshold = 2
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(read_path) as fr:
        metas = json.load(fr)
    print(f"Total {len(metas)} samples")

    num_begin = 0
    num_compare_start = 0
    num_compare_end = 0
    num_than = 0
    num_in = 0
    num_of = 0
    num_from = 0
    num_other = 0

    metas_ref = []
    num = len(metas)
    for idx in range(num - 1, -1, -1):
        meta = metas[idx]
        degradation = meta["distortion_class"]
        severity = meta["severity"]
        if exclude and not check_include(degradation, severity, threshold):
            del metas[idx]
            continue

        meta["image_ref"] = None
        meta["image_B"] = None
        meta["task_type"] = "quality_single_A_noref"

        if training:
            resp = meta["conversations"][1]["value"]

            if meta["distortion_class"] is None:
                resp = resp.replace("the reference image", "a high-quality image")
                resp = resp.replace("the reference", "a high-quality image")
                resp = resp.replace(
                    "the high-quality reference", "a high-quality image"
                )
                resp = resp.replace("a reference standard", "a high-quality image")
                assert not "reference" in resp, f"{resp}"

            if resp.startswith("The reference image"):
                num_begin += 1
                resp = resp.replace("The reference image", "The image")

            if (
                "Compared to the reference image, " in resp
                or "Compared to the reference, " in resp
            ):
                num_compare_start += 1
                idx = resp.find("Compared to the reference image, ")
                if idx >= 0:
                    idx = idx + len("Compared to the reference image, ")
                    resp = resp[:idx] + resp[idx].upper() + resp[idx + 1 :]
                idx = resp.find("Compared to the reference, ")
                if idx >= 0:
                    idx = idx + len("Compared to the reference, ")
                    resp = resp[:idx] + resp[idx].upper() + resp[idx + 1 :]
                resp = resp.replace("Compared to the reference image, ", "")
                resp = resp.replace("Compared to the reference, ", "")

            if (
                " when compared to the reference image" in resp
                or " compared to the reference image" in resp
                or " when compared to the reference" in resp
                or " compared to the reference" in resp
            ):
                num_compare_end += 1
                resp = resp.replace(" when compared to the reference image", "")
                resp = resp.replace(" compared to the reference image", "")
                resp = resp.replace(" when compared to the reference", "")
                resp = resp.replace(" compared to the reference", "")

            if " than the reference image" in resp or " than the reference" in resp:
                num_than += 1
                resp = resp.replace(" than the reference image", "")
                resp = resp.replace(" than the reference", "")

            if " in the reference image" in resp or " in the reference" in resp:
                num_in += 1
                resp = resp.replace(" in the reference image", "")
                resp = resp.replace(" in the reference", "")

            if " of the reference image" in resp or " of the reference" in resp:
                num_of += 1
                resp = resp.replace(" of the reference image", "")
                resp = resp.replace(" of the reference", "")

            if " from the reference image" in resp or " from the reference" in resp:
                num_from += 1
                resp = resp.replace(" from the reference image", "")
                resp = resp.replace(" from the reference", "")

            if (
                "the reference image" in resp
                or "the reference" in resp
                or "a reference image" in resp
                or "a reference" in resp
                or "reference image" in resp
                or " reference." in resp
                or " reference," in resp
            ):
                num_other += 1
                resp = resp.replace("the reference image", "a high-quality image")
                resp = resp.replace("the reference", "a high-quality image")
                resp = resp.replace("a reference image", "a high-quality image")
                resp = resp.replace("a reference", "a high-quality image")
                resp = resp.replace("reference image", "high-quality image")
                resp = resp.replace(" reference.", " high-quality image.")
                resp = resp.replace(" reference,", " high-quality image,")

            meta["conversations"][1]["value"] = resp
            if " reference" in resp:
                print("=" * 100)
                print(resp)
                metas_ref.append(meta)

    if training:
        print(f"Total {len(metas_ref)} samples with reference")
        print("=" * 100)
        print("num_begin", num_begin)
        print("num_compare_start", num_compare_start)
        print("num_compare_end", num_compare_end)
        print("num_than", num_than)
        print("num_in", num_in)
        print("num_of", num_of)
        print("num_from", num_from)
        print("num_other", num_other)

    with open(save_path, "w") as fw:
        print(f"{len(metas)} samples saved")
        fw.write(json.dumps(metas, indent=4))
