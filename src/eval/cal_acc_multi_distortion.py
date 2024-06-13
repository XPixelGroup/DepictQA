import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="evaluation parameters for DepictQA")
    parser.add_argument("--pred_path", type=str, default=None)
    parser.add_argument("--gt_path", type=str, default=None)
    parser.add_argument("--confidence", action="store_true")
    parser.add_argument(
        "--intervals",
        type=float,
        nargs="+",
        default=[0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1],
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    with_confidence = args.confidence
    intervals = args.intervals
    if not with_confidence:
        intervals = [0, 1]
    num_all = [0] * (len(intervals) - 1)
    num_right = [0] * (len(intervals) - 1)

    # load predict results
    pred_path = args.pred_path
    if pred_path.endswith(".jsonl"):
        pred_metas = []
        with open(pred_path) as fr:
            for line in fr:
                pred_meta = json.loads(line)
                pred_metas.append(pred_meta)
    else:
        assert pred_path.endswith(".json")
        with open(pred_path) as fr:
            pred_metas = json.load(fr)

    # load gt results
    with open(args.gt_path) as fr:
        gt_metas = json.load(fr)

    for pred_meta in pred_metas:
        for gt_meta in gt_metas:
            if pred_meta["id"] == gt_meta["id"]:
                confidence = pred_meta["confidence"]
                # round to avoid float error: confidence can slightly > 1
                confidence = round(confidence, 6) if confidence else confidence
                if not with_confidence:
                    # if not with_confidence, give all samples same confidence
                    confidence = 0.5

                idx = 0
                while True:
                    if confidence > intervals[idx] and confidence <= intervals[idx + 1]:
                        break
                    idx += 1

                num_all[idx] += 1
                pred = [
                    _.strip().lower() for _ in pred_meta["text"].strip().split("and")
                ]
                pred = sorted(list(set(pred)))
                gt = sorted(
                    [_.strip().lower() for _ in gt_meta["answer"].strip().split("and")]
                )
                if pred == gt:
                    num_right[idx] += 1
                else:
                    for pred_ in pred:
                        if pred_ in gt:
                            num_right[idx] += 1 / max(len(pred), len(gt))
                break

    for idx in range(len(num_all)):
        interval = f"[{intervals[idx]},{intervals[idx + 1]}]"
        if num_all[idx] == 0:
            print(f"Confidence Interval {interval}, {num_all[idx]} Samples")
        else:
            acc = round(num_right[idx] / num_all[idx], 4)
            print(
                f"Confidence Interval {interval}, {num_all[idx]} Samples, Accuracy: {acc}"
            )

    acc = round(sum(num_right) / sum(num_all), 4)
    print(f"Average Accuracy: {acc}")
