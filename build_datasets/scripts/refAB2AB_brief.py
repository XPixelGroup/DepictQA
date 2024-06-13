import argparse
import json

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--read_path", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    read_path = args.read_path
    save_path = args.save_path
    with open(read_path) as fr:
        metas = json.load(fr)
    for meta in metas:
        meta["image_ref"] = None
        meta["task_type"] = "quality_compare_noref"
    with open(save_path, "w") as fw:
        fw.write(json.dumps(metas, indent=4))
