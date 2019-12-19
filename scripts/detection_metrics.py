#!/usr/bin/env python3

""" Generate statistics on detections given a truth file + a detect inference output """

import argparse
import pandas as pd
import progressbar
import numpy as np

def _rowToBoxDict(row):
    """ Converts a row from a csv to a dictionary """
    box_dict={}
    box_dict['x'] = row.x
    box_dict['y'] = row.y
    # Support both row formats for detect versus truth
    try:
        box_dict['w'] = row.width
        box_dict['h'] = row.height
    except:
        box_dict['w'] = row.w
        box_dict['h'] = row.h
    return box_dict
def _intersection_over_union(boxA, boxB):
    """ Computes intersection over union for two bounding boxes.
        Inputs:
        boxA -- First box. Must be a dict containing x, y, w, h.
        boxB -- Second box. Must be a dict containing x, y, w, h.
        Return:
        Intersection over union.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(int(boxA["x"]), int(boxB["x"]))
    yA = max(int(boxA["y"]), int(boxB["y"]))
    xB = min(int(boxA["x"]) + int(boxA["w"]), int(boxB["x"]) + int(boxB["w"]))
    yB = min(int(boxA["y"]) + int(boxA["h"]), int(boxB["y"]) + int(boxB["h"]))

    # compute the area of intersection rectangle
    interX = xB - xA + 1
    interY = yB - yA + 1
    if interX < 0 or interY < 0:
        iou = 0.0
    else:
        interArea = float((xB - xA + 1) * (yB - yA + 1))
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = int(boxA["w"]) * int(boxA["h"])
        boxBArea = int(boxB["w"]) * int(boxB["h"])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        if float(boxAArea + boxBArea - interArea) <= 0.0:
            return 0.01
        try:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        except:
            print("interArea: {}".format(interArea))
            print("Union: {}".format(float(boxAArea + boxBArea - interArea)))
        # return the intersection over union value
    return iou

def calculateStats(truth, detections, keep_threshold):
    eval_detects = detections.loc[detections.det_conf > keep_threshold]
    count = len(eval_detects)
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    double_counts = 0

    matches={}
    # Iterate over all detections from inference over keep threshold
    # calculate true and false positives
    for idx, row in eval_detects.iterrows():
        matching_truth_df = truth.loc[(truth.video_id == row.video_id) & (truth.frame == row.frame)]
        if len(matching_truth_df) == 0:
            false_positives += 1
        else:
            got_match = False
            for truth_idx, truth_row in matching_truth_df.iterrows():
                truth_box = _rowToBoxDict(truth_row)
                canidate_box = _rowToBoxDict(row)
                iou = _intersection_over_union(truth_box, canidate_box)
                if iou > args.iou_threshold:
                    got_match = True
                    if truth_row.name in matches:
                        double_counts += 1
                    else:
                        matches[truth_row.name] = row
                    break

            if got_match == True:
                true_positives += 1
            else:
                false_positives += 1

    counted=[]
    for idx, row in truth.iterrows():
        matching_detection_df = eval_detects.loc[(eval_detects.video_id == row.video_id) & (eval_detects.frame == row.frame)]
        boxes_in_truth=len(truth.loc[(truth.video_id == row.video_id) & (truth.frame == row.frame)])
        boxes_in_detection = len(matching_detection_df)
        if boxes_in_detection < boxes_in_truth:
            vid_tag=f"{row.video_id}_{row.frame}"
            if not vid_tag in counted:
                counted.append(vid_tag)
                false_negatives += (boxes_in_truth - boxes_in_detection)

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    return (precision, recall, double_counts / true_positives)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth")
    parser.add_argument("--keep-threshold-min", type=float, default=0.05)
    parser.add_argument("--keep-threshold-max", type=float, default=0.80)
    parser.add_argument("--keep-threshold-steps", type=int, default=10)
    parser.add_argument("--iou-threshold", type=float, default=0.4)
    parser.add_argument("--output-matrix", type=str)
    parser.add_argument("detect_csv")
    args = parser.parse_args()

    detections=pd.read_csv(args.detect_csv)
    truth = pd.read_csv(args.truth)
    results=[]
    keep_thresholds = np.linspace(args.keep_threshold_min,
                                  args.keep_threshold_max,
                                  args.keep_threshold_steps)
    bar = progressbar.ProgressBar(redirect_stdout=True)
    for keep_threshold in bar(keep_thresholds):
        result_tuple = calculateStats(truth, detections, keep_threshold)
        results.append([keep_threshold, *result_tuple])

    matrix = np.array(results)
    print(matrix)
    if args.output_matrix:
        np.save(args.output_matrix, matrix)
