import argparse
import base64
import glob
import json
import os

from openai import OpenAI

API_KEY = ""

parser = argparse.ArgumentParser(description="Test X-Distortion")
parser.add_argument("--meta_dir", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--fail_dir", type=str, required=True)


def encode_img(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def gpt4v(img_ref_path, img_A_path, distortion_class, severity):
    add_distortion = True
    if severity == 0:
        assert distortion_class is None
        add_distortion = False

    grades = ["slight", "moderate", "obvious", "serious", "catastrophic"]
    grades_str = "[slight, moderate, obvious, serious, catastrophic]"
    img_ref_base64 = encode_img(img_ref_path)
    img_A_base64 = encode_img(img_A_path)

    if add_distortion:
        grade = grades[severity - 1]
        query = (
            "You are an expert in image quality assessment. "
            + "The first image is a reference image, and the second image is the image to be evaluated. "
            + "The evaluated image is generated by adding distortion into the reference. "
            + f"The added distortion is {distortion_class} with grade {grade} (out of {grades_str}). "
            + "Please assess the quality of the evaluated image. "
            + "The response should cover three areas. "
            + "First, a short description of the image content. "
            + "Second, distortion identification in the evaluated image and discussion on how this distortion affects the image content. "
            + "Third, a brief summary of the overall quality of the evaluated image. "
            + "The response must not show that you were given the reference. "
            + "The whole response must be below 100 words."
        )
    else:
        query = (
            "You are an expert in image quality assessment. "
            + "The first image is a reference image, and the second image is an image to be evaluated. "
            + "The evaluated image is a high quality image with no distortions, the same as the reference. "
            + "Please assess the quality of the evaluated image. "
            + "The response should cover three areas. "
            + "First, a short description of the image content. "
            + "Second, a detailed description of the evaluated image's quality. "
            + "Third, a brief summary of the overall quality of the evaluated image. "
            + "The response must not show that you were given the reference. "
            + "The whole response must be below 100 words."
        )

    client = OpenAI(api_key=API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpg;base64,{img_ref_base64}",
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpg;base64,{img_A_base64}",
                    },
                ],
            }
        ],
        temperature=0.5,
        max_tokens=200,
    )
    content = resp.choices[0].message.content
    return content


if __name__ == "__main__":
    args = parser.parse_args()
    idx_meta_start = 0
    idx_meta_end = 5

    meta_dir = args.meta_dir
    meta_paths = sorted(glob.glob(os.path.join(meta_dir, "*.json")))
    save_dir = args.save_dir
    fail_dir = args.fail_dir
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    distortion_paths_error = []
    for idx_meta, meta_path in enumerate(meta_paths[idx_meta_start:idx_meta_end]):
        print("=" * 100)
        print(idx_meta + idx_meta_start)

        meta_name = os.path.basename(meta_path)
        save_path = os.path.join(save_dir, meta_name)
        if os.path.exists(save_path):
            print(f"{save_path} has been generated, skip.")
            continue

        with open(meta_path) as fr:
            meta = json.load(fr)
        ref_path = meta["img_ref"]
        dist_path = meta["img_lq"]
        distortion_class = meta["distortion_class"]
        distortion_name = meta["distortion_name"]
        severity = meta["severity"]
        if severity == 0:
            assert distortion_class is None and distortion_name is None
        try:
            content = gpt4v(ref_path, dist_path, distortion_class, severity)
            meta["text"] = content
            with open(save_path, "w") as fw:
                fw.write(json.dumps(meta, indent=4))
            print(meta)
        except:
            import sys

            except_type, except_value, except_traceback = sys.exc_info()
            except_file = os.path.split(except_traceback.tb_frame.f_code.co_filename)[1]
            exc_dict = {
                "error type": except_type,
                "error info": except_value,
                "error file": except_file,
                "error line": except_traceback.tb_lineno,
            }
            print(exc_dict)
            distortion_paths_error.append(dist_path)

    fail_path = os.path.join(fail_dir, "res_fail.txt")
    with open(fail_path, "w") as fw:
        fw.write("\n".join(distortion_paths_error))
