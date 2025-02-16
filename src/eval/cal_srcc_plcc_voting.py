import argparse
import json

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr


def calculate_srcc(pred, mos):
    srcc, _ = spearmanr(pred, mos)
    return srcc


def calculate_plcc(pred, mos):
    plcc, _ = pearsonr(pred, mos)
    return plcc


def fit_curve(x, y, curve_type='logistic_4params'):
    r'''Fit the scale of predict scores to MOS scores using logistic regression suggested by VQEG.
    The function with 4 params is more commonly used.
    The 5 params function takes from DBCNN:
        - https://github.com/zwx8981/DBCNN/blob/master/dbcnn/tools/verify_performance.m
    '''
    assert curve_type in [
        'logistic_4params', 'logistic_5params'], f'curve type should be in [logistic_4params, logistic_5params], but got {curve_type}.'

    betas_init_4params = [np.max(y), np.min(y), np.mean(x), np.std(x) / 4.]

    def logistic_4params(x, beta1, beta2, beta3, beta4):
        yhat = (beta1 - beta2) / (1 + np.exp(- (x - beta3) / beta4)) + beta2
        return yhat

    betas_init_5params = [10, 0, np.mean(y), 0.1, 0.1]

    def logistic_5params(x, beta1, beta2, beta3, beta4, beta5):
        logistic_part = 0.5 - 1. / (1 + np.exp(beta2 * (x - beta3)))
        yhat = beta1 * logistic_part + beta4 * x + beta5
        return yhat

    if curve_type == 'logistic_4params':
        logistic = logistic_4params
        betas_init = betas_init_4params
    elif curve_type == 'logistic_5params':
        logistic = logistic_5params
        betas_init = betas_init_5params

    betas, _ = curve_fit(logistic, x, y, p0=betas_init, maxfev=10000)
    yhat = logistic(x, *betas)
    return yhat


def parse_args():
    parser = argparse.ArgumentParser(description="evaluation parameters for DepictQA")
    parser.add_argument("--test_paths", type=str, required=True, nargs="+")
    parser.add_argument("--pred_paths", type=str, required=True, nargs="+")
    parser.add_argument("--gt_paths", type=str, required=True, nargs="+")
    parser.add_argument("--confidence", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    confidence = args.confidence
    test_paths = args.test_paths
    pred_paths = args.pred_paths
    gt_paths = args.gt_paths

    for test_path, pred_path, gt_path in zip(test_paths, pred_paths, gt_paths):
        with open(test_path) as fr:
            test_metas = json.load(fr)
        test_metas = sorted(test_metas, key=lambda x: x["id"])

        with open(pred_path) as fr:
            pred_metas = json.load(fr)
        pred_metas = sorted(pred_metas, key=lambda x: x["id"])

        vote_dict = {}
        for test_meta, meta in zip(test_metas, pred_metas):
            assert test_meta["id"] == meta["id"]
            img_ref = test_meta["image_ref"]
            img_A = test_meta["image_A"]
            img_B = test_meta["image_B"]
            if meta["text"].strip() == "Image A":
                score_A = 1 if not confidence else meta["confidence"]
                score_B = 1 - score_A
            else:
                assert meta["text"].strip() == "Image B"
                score_B = 1 if not confidence else meta["confidence"]
                score_A = 1 - score_B

            if img_A not in vote_dict:
                vote_dict[img_A] = {
                    "score": score_A,
                    "count": 1
                    }
            else:
                vote_dict[img_A]["score"] += score_A
                vote_dict[img_A]["count"] += 1
            if img_B not in vote_dict:
                vote_dict[img_B] = {
                    "score": score_B,
                    "count": 1
                    }
            else:
                vote_dict[img_B]["score"] += score_B
                vote_dict[img_B]["count"] += 1

        pred_dict = {}
        for key in vote_dict:
            pred_dict[key] = vote_dict[key]["score"] / vote_dict[key]["count"]

        with open(gt_path) as fr:
            gt_dict = json.load(fr)

        srcc_list = []
        plcc_list = []
        for key in gt_dict:
            gt_data = gt_dict[key]
            dist_paths = gt_data["dist_paths"]
            gts = gt_data["mos_list"]
            preds = [pred_dict[_] for _ in dist_paths]
            preds_fit = fit_curve(preds, gts)
            srcc = calculate_srcc(preds_fit, gts)
            plcc = calculate_plcc(preds_fit, gts)
            srcc_list.append(srcc)
            plcc_list.append(plcc)

        srcc = sum(srcc_list) / len(srcc_list)
        plcc = sum(plcc_list) / len(plcc_list)

        print("=" * 100)
        print("Pred: ", pred_path)
        print("SRCC: ", srcc)
        print("PLCC: ", plcc)
