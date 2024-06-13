import argparse
import json
import os

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--read_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--exclude", action="store_true")


def check_include(degradations, severities, threshold):
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
    if not degradations:
        assert severities == 0
        return True
    for degradation, severity in zip(degradations, severities):
        for distortion in distortions_thre:
            if distortion in degradation and severity < threshold:
                return False
    return True


if __name__ == "__main__":
    args = parser.parse_args()
    read_path = args.read_path
    save_path = args.save_path
    exclude = args.exclude

    threshold = 2
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(read_path) as fr:
        metas = json.load(fr)
    print(f"Total {len(metas)} samples")

    num = len(metas)
    for idx in range(num - 1, -1, -1):
        meta = metas[idx]
        degradations_A = meta["distortions_A"]["distortion_classes"]
        severities_A = meta["distortions_A"]["severities"]
        degradations_B = meta["distortions_B"]["distortion_classes"]
        severities_B = meta["distortions_B"]["severities"]
        if exclude and not (
            check_include(degradations_A, severities_A, threshold)
            and check_include(degradations_A, severities_A, threshold)
        ):
            del metas[idx]
            continue

    for meta in metas:
        meta["image_ref"] = None
        meta["task_type"] = "quality_compare_noref"

    with open(save_path, "w") as fw:
        print(f"{len(metas)} samples saved")
        fw.write(json.dumps(metas, indent=4))
