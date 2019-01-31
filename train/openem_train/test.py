__copyright__ = "Copyright (C) 2018 CVision AI."
__license__ = "GPLv3"
# This file is part of OpenEM, released under GPLv3.
# OpenEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

import os
import sys
import math
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
sys.path.append('../python')
import openem

"""Functions for testing end to end model.
"""
def _find_roi(mask_finder_path, vid_path):
    """Finds ROI in a video.

    # Arguments
        mask_finder_path: Path to find_ruler model file.
        vid_path: Path to the video.

    # Returns:
        Region of interest and ruler endpoints.

    # Raises:
        IOError: If video or model file cannot be opened.
        RuntimeError: If fails to add images or process model.
    """
    # Determined by experimentation with GPU having 8GB memory.
    max_img = 8

    # Create and initialize the mask finder.
    mask_finder = openem.RulerMaskFinder()
    status = mask_finder.Init(mask_finder_path)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize mask finder!")

    # Set up video reader.
    reader = openem.VideoReader()
    status = reader.Init(vid_path)
    if not status == openem.kSuccess:
        raise IOError("Failed to open video!")

    # Decode the first 100 frames and take the average mask.
    mask_avg = np.zeros((reader.Height(), reader.Width()))
    num_masks = 0
    masks = openem.VectorImage()
    vid_end = False
    for i in range(math.ceil(100 / max_img)):
        for j in range(max_img):
            img = openem.Image()
            status = reader.GetFrame(img)
            if not status == openem.kSuccess:
                vid_end = True
                break
            status = mask_finder.AddImage(img)
            if not status == openem.kSuccess:
                raise RuntimeError("Failed to add frame to mask finder!")
        status = mask_finder.Process(masks)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to process mask finder!")
        for mask in masks:
            mask.Resize(reader.Width(), reader.Height())
            mask_data = mask.DataCopy()
            mask_data = np.array(mask_data)
            mask_data = np.reshape(mask_data, (reader.Height(), reader.Width()))
            mask_avg += mask_data
            num_masks += 1
        if vid_end:
            break

    # Convert mean mask from numpy to openem format.
    mask_vec = mask_avg.copy()
    mask_vec = mask_vec / np.max(mask_vec)
    mask_vec = mask_vec * 255.0
    mask_vec = mask_vec.reshape(-1).astype(np.uint8).tolist()
    mask_img = openem.Image()
    mask_img.FromData(mask_vec, reader.Width(), reader.Height(), 1)

    # Now that we have the best mask, use this to compute the ROI.
    endpoints = openem.RulerEndpoints(mask_img)
    r_mask = openem.Rectify(mask_img, endpoints)
    roi = openem.FindRoi(r_mask)
    return (roi, endpoints)

def _detect_and_classify(detect_path, classify_path, vid_path, roi, endpoints):
    """Finds and classifies detections for all frames in a video.

    # Arguments
        detect_path: Path to detect model file.
        classify_path: Path to classify model file.
        vid_path: Path to the video.
        roi: Region of interest output from find_roi.
        endpoints: Ruler endpoints from find_roi.

    # Returns
        Detection rects and classification scores.

    # Raises
        IOError: If video or model files cannot be opened.
        RuntimeError: If unable to add frame or process a model.
    """
    # Determined by experimentation with GPU having 8GB memory.
    max_img = 32

    # Create and initialize the detector.
    detector = openem.Detector()
    status = detector.Init(detect_path, 0.5)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize detector!")

    # Create and initialize the classifier.
    classifier = openem.Classifier()
    status = classifier.Init(classify_path, 0.5)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize classifier!")

    # Initialize the video reader.
    reader = openem.VideoReader()
    status = reader.Init(vid_path)
    if not status == openem.kSuccess:
        raise IOError("Failed to open video {}!".format(vid_path))

    # Iterate through frames.
    vid_end = False
    detections = []
    scores = []
    while True:

        # Find detections.
        dets = openem.VectorVectorDetection()
        imgs = [openem.Image() for _ in range(max_img)]
        for i, img in enumerate(imgs):
            status = reader.GetFrame(img)
            if not status == openem.kSuccess:
                vid_end = True
                break
            img = openem.Rectify(img, endpoints)
            img = openem.Crop(img, roi)
            status = detector.AddImage(img)
            imgs[i] = img
            if not status == openem.kSuccess:
                raise RuntimeError("Failed to add frame to detector!")
        status = detector.Process(dets)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to process detector!")
        detections += dets

        # Classify detections
        for det_frame, img in zip(dets, imgs):
            score_batch = openem.VectorClassification()
            for det in det_frame:
                det_img = openem.GetDetImage(img, det.location)
                status = classifier.AddImage(det_img)
                if not status == openem.kSuccess:
                    raise RuntimeError("Failed to add frame to classifier!")
            status = classifier.Process(score_batch)
            if not status == openem.kSuccess:
                raise RuntimeError("Failed to process classifier!")
            scores.append(score_batch)
        if vid_end:
            break
    return (detections, scores)

def _write_counts(
        count_path,
        out_path,
        roi,
        detections,
        scores,
        species_list,
        vid_width,
        detect_width):
    """Writes a csv file containing fish species and frame numbers.

    # Arguments
        count_path: Path to count model file.
        out_path: Path to output csv file.
        roi: Region of interest, needed for image width and height.
        detections: Detections for each frame.
        scores: Cover and species scores for each detection.
        species_list: List of species.
        vid_width: Width of the original video.
    """
    # Create and initialize keyframe finder.
    finder = openem.KeyframeFinder()
    status = finder.Init(count_path, roi[2], roi[3])
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize keyframe finder!")

    # Process keyframe finder.
    keyframes = openem.VectorInt()
    status = finder.Process(scores, detections, keyframes)
    if not status == openem.kSuccess:
        msg = "Failed to process keyframe finder! Error code {}"
        raise RuntimeError(msg.format(status))

    # Write the keyframes out.
    with open(out_path, "w") as csv:
        csv.write("frame,species,length\n")
        for i in keyframes:
            c = scores[i][0]
            max_score = 0.0
            species_index = 0
            for j, s in enumerate(c.species):
                if j == 0:
                    continue
                if s > max_score:
                    max_score = s
                    species = species_list[j]
            d = detections[i][0]
            _, _, length, _ = d.location
            length *= float(detect_width) / float(vid_width)
            csv.write("{},{},{}\n".format(i, species, length))

def _losses(pred, truth, species):
    """Computes the losses between two sequences of keyframes.

    # Arguments:
        pred: N by 3 pandas data frame containing frame, species, length.
        truth: N by 3 pandas data frame containing frame, species, length.
        species: List of species names for confusion matrix.

    # Returns:
        Tuple containing:
        - Number of true positives
        - Number of false positives
        - Number of false negatives
        - Number of correct classifications (for true positives only)
        - Sum of absolute differences in length (for true positives only)
        - Confusion matrix (for true positives only)
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    num_correct = 0
    sum_abs_diff = 0.0
    num_classes = len(species)
    confusion = np.zeros((num_classes, num_classes))

    # Map between truth index (key) and pred indices (values).
    mapping = defaultdict(list)
    for p_idx, p_row in pred.iterrows():
        t_idx = (truth['frame'] - p_row.frame).abs().argsort()[0]
        mapping[t_idx].append((p_row.frame, p_row.species, p_row.length))

    # Iterate through map and compute statistics.
    for t_idx, _ in truth.iterrows():
        matches = mapping[t_idx]
        t_row = truth.iloc[t_idx]
        if matches:
            true_pos += 1
            false_pos += len(matches) - 1
            matches.sort(key=lambda x: abs(x[0] - t_row.frame))
            truth_species = t_row.species.lower()
            pred_species = matches[0][1].lower()
            if truth_species not in species:
                truth_species = 'other'
            confusion[
                species.index(truth_species),
                species.index(pred_species)
            ] += 1
            if pred_species == truth_species:
                num_correct += 1
            sum_abs_diff += abs(matches[0][2] - t_row.length)
        else:
            false_neg += 1
    return (true_pos, false_pos, false_neg, num_correct, sum_abs_diff, confusion)

def predict(config):

    # Get paths from config file.
    videos = config.test_vids()
    out_dir = config.test_output_dir()
    find_ruler_path = config.find_ruler_model_path()
    detect_path = config.detect_model_path()
    classify_path = config.classify_model_path()
    count_path = config.count_model_path()
    os.makedirs(out_dir, exist_ok=True)

    # Iterate through videos.
    for vid in videos:

        # Get the video width.
        reader = openem.VideoReader()
        status = reader.Init(vid)
        if not status == openem.kSuccess:
            raise IOError("Failed to read video {}!".format(vid))
        vid_width = float(reader.Width())

        # Get path to output csv.
        _, fname = os.path.split(vid)
        base, _ = os.path.splitext(fname)
        out_path = os.path.join(out_dir, base + '.csv')

        # Find the ROI.
        print("Doing prediction on {}...".format(vid))
        roi, endpoints = _find_roi(find_ruler_path, vid)

        # Do detection and classification.
        detections, scores = _detect_and_classify(
            detect_path,
            classify_path,
            vid,
            roi,
            endpoints
        )
        # Write counts to csv.
        _write_counts(
            count_path,
            out_path,
            roi,
            detections,
            scores,
            ['unknown',] + config.species(),
            vid_width,
            config.detect_width()
        )

def eval(config):

    # Get paths from config file.
    test_dir = config.test_output_dir()
    truth_files = config.test_truth_files()

    # Initialize evaluation metrics.
    species = config.species()
    species = [s.lower() for s in species]
    true_pos = 0
    false_pos = 0
    false_neg = 0
    num_correct = 0
    sum_abs_diff = 0.0
    confusion = np.zeros((len(species), len(species)))

    # Iterate through truth/test files and sum up the metrics.
    for truth_file in truth_files:
        _, fname = os.path.split(truth_file)
        test_file = os.path.join(test_dir, fname)
        if os.path.exists(test_file):
            print("Evaluating performance with {}...".format(test_file))
            truth = pd.read_csv(truth_file)
            pred = pd.read_csv(test_file)
            metrics = _losses(pred, truth, species)
            true_pos += metrics[0]
            false_pos += metrics[1]
            false_neg += metrics[2]
            num_correct += metrics[3]
            sum_abs_diff += metrics[4]
            confusion += metrics[5]
        else:
            msg = "Could not find test output {}! Excluding from evaluation..."
            print(msg.format(test_file))

    # Compute summary statistics.
    print("True positives: {}".format(true_pos))
    print("False positives: {}".format(false_pos))
    print("False negatives: {}".format(false_neg))
    true_pos = float(true_pos)
    false_pos = float(false_pos)
    false_neg = float(false_neg)
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1 = 2 * precision * recall / (precision + recall)
    print("Precision: {0:.4f}".format(precision))
    print("Recall: {0:.4f}".format(recall))
    print("F1 score: {0:.4f}".format(f1))
    class_acc = float(num_correct) / true_pos
    print("Classification accuracy: {0:.4f}".format(class_acc))
    np.set_printoptions(suppress=True)
    print("Confusion matrix:")
    print("{}".format(confusion))
    mean_abs_diff = sum_abs_diff / true_pos
    print("Mean absolute length difference: {0:.4f}".format(mean_abs_diff))
