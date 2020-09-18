#!/usr/bin/env python

import argparse
import os
import sys
import numpy as np
from pprint import pprint
import progressbar
from utilities import is_valid_path
from utilities import ModelData
from utilities import TrackData
from utilities import get_image
import functools 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Displays pairs of images and reidentification " +
        "output.")
    parser.add_argument("data_dir",
        type=lambda x: is_valid_path(parser, x),
        help="Directory containing reidentification data.")
    parser.add_argument("--data",
        type=str,
        default="validate",
        help="Which data to display, must be 'training' or 'validate'.")
    args = parser.parse_args()
    model_data = ModelData(args.data_dir)
    mean_img = model_data.mean_image()
    extractor = model_data.appearance_extractor()
    comparator = model_data.appearance_comparator()
    if args.data == "validate":
        test_data = model_data.cnn_validate()
    elif args.data == "training":
        test_data = model_data.cnn_training()
    dir_idx_list = []
    idx_list = []
    tdata_list = [TrackData(os.path.join(args.data_dir,d)) for d in model_data.track_dirs()]
    for dir_idx, track_data in enumerate(tdata_list):
        num_img = track_data.num_detection_images()
        dir_idx_list += [dir_idx for _ in range(num_img)]
        idx_list += range(num_img)
    mapping = np.transpose([dir_idx_list, idx_list])
    # Bind the track data list and index mapping to image accessor.
    get_img_bound = functools.partial(get_image, tdata_list, mapping)
    score = {0: {True: 0, False: 0}, 1: {True: 0, False: 0}}
    bar = progressbar.ProgressBar(redirect_stdout=True)
    for index, label in bar(zip(test_data["index"], test_data["outputs"])):
        img0 = get_img_bound(index[0])
        img1 = get_img_bound(index[1])
        fea0 = extractor(img0)
        fea1 = extractor(img1)

        dist = comparator.predict([fea0,fea1])[0][0]
        print(f"{index} : {dist} vs. {label}")
        correct = round(dist) == label[0]
        score[label[0]][correct] += 1
    pprint(score)
