#!/usr/bin/env python3

""" Generate statistics on detections given a truth file + a detect inference output

This script sweeps across different keep thresholds based on the provided
CLI arguments (`--keep-threshold-[min,max,steps]`) and generates a
precision/recall graph. It also calculates 'double count' metric which is
the number of boxes that matched a truth box, but already had a box associated
with it. Imagine two boxes around the same object, with slightly different
confidences.

The True Positive is based on the IoU of the detection box against the truth
data. If the detection box is not within the IoU threshold of the truth it is
counted as a false positive.

The false negatives are calculated per frame, such that if a frame has 4 truth
detections, but 2 inference detections, 2 false negatives are added to the
metric.

"""

import argparse
import pandas as pd
import progressbar
import numpy as np
from tqdm import tqdm
import math

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
    image_names = truth['video_id'].unique()
    matches={}
    # Iterate over all detections from inference over keep threshold
    # calculate true and false positives
    for image_name in tqdm(image_names):
        img_detection = eval_detects[eval_detects['video_id'] == image_name]
        img_truth = truth[truth['video_id'] == image_name]
        for idx, row in img_detection.iterrows():
            matching_truth_df = img_truth.loc[(img_truth.frame == row.frame)]
            if len(matching_truth_df) == 0:
                false_positives += 1
            else:
                got_match = False
                for truth_idx, truth_row in matching_truth_df.iterrows():
                    truth_box = _rowToBoxDict(truth_row)
                    if not math.isnan(truth_box['x']):
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
        for idx, row in img_truth.iterrows():
            if not math.isnan(row['x']):
                truth_box = _rowToBoxDict(row)
                match_found = False
                for idx, det in img_detection.iterrows():
                    det_box = _rowToBoxDict(det)
                    iou = _intersection_over_union(truth_box, det_box)
                    if iou > 0.4:
                        match_found = True
                        break
                if not match_found:
                    false_negatives += 1
                    
    all_true_positives = true_positives
    true_positives -=  double_counts
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    return (precision, recall, double_counts / all_true_positives)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth", help="Truth CSV file")
    parser.add_argument("--keep-threshold-min",
                        type=float,
                        default=0.05,
                        help="Minimum keep threshold to scan")
    parser.add_argument("--keep-threshold-max",
                        type=float,
                        default=0.80,
                        help="Maximum keep threshold to scan")
    parser.add_argument("--keep-threshold-steps",
                        type=int,
                        default=10,
                        help="Number of steps to use between min/max")
    parser.add_argument("--iou-threshold",
                        type=float,
                        default=0.4,
                        help="IoU threshold for determining True Positive")
    parser.add_argument("--output-matrix",
                        type=str,
                        help="If supplied, dumps the matrix to a file, else just prints")
    parser.add_argument("detect_csv", help="Inference result CSV")
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
