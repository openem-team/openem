#!/usr/bin/env python

import argparse
import json
"""
Connects to tator and generates a training folder (model_dir) with the following contents:

.. code-block:: bash

    track_dirs.txt - A list of directories containing tracks
    tracks - folder for tracks
       |
       00001 - Track 1 folder
         |
         |- detection_images
         |    |
         |    |-00000001.png
         |    |-00000002.png
       0000N - Track N folder

`track_dirs.txt` can be either full paths or relative to model dir. In the example above track_dirs.txt would contain:

.. code-block:: bash

   tracks/00001
        ...
   tracks/0000N
"""
import cv2
import os
import numpy as np
import pytator
from utilities import is_valid_path
from utilities import ModelData
from utilities import TrackData
from utilities import get_session
import progressbar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates training dataset for reidentification CNN.  " +
            "The output file will contain two arrays: index " +
            "and outputs.  index contains a num examples by 2 " +
            "array of indices into the imgs array.  outputs contains " +
            "labels corresponding to each example.")
    pytator.tator.cli_parser(parser)
    parser.add_argument("--track-type-id", required=True, type=int)
    parser.add_argument("--holdout-fraction",
                        type=float,
                        default=0.1,
                        help="Fraction of data to be stored in a separate file for " + "validation.")
    parser.add_argument("--max_frame_diff", type=int, default=-1)
    parser.add_argument("model_dir",
        help="Path to model directory.",
        default=os.getcwd())
    args = parser.parse_args()
    get_session()
    model_data = ModelData(args.model_dir)
    tracks_dir = os.path.join(args.model_dir, "tracks")
    os.makedirs(tracks_dir, exist_ok=True)
    tator = pytator.Tator(args.url, args.token, args.project)
    # Generate pair indices
    index0 = []
    index1 = []
    label = []
    offset = 0

    track_dirs_fp=open(os.path.join(args.model_dir, "track_dirs.txt"), 'w')

    # TODO: Only get the tracks and detections with the right frame diff
    tracks = tator.State.filter({"type": args.track_type_id})
    print(f"Processing {len(tracks)} Tracks")
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(tracks))
    for idx, track in bar(enumerate(tracks)):
        track_id = track['id']
        if track['attributes']['Match'].find("Bad") >= 0:
            print(f"Skipping bad track, {track['id']}")
            continue

        if track['attributes']['Match'].find("Unknown") >= 0:
            print(f"Skipping unknown track, {track['id']}")
            continue

        # Create a folder based on the database id number
        this_track_dir=os.path.join(tracks_dir, f"{track_id:09d}")
        os.makedirs(this_track_dir, exist_ok=True)
        track_data = TrackData(this_track_dir)

        # Output track info
        with open(os.path.join(this_track_dir, "track.json"), 'w') as fp:
            json.dump(track, fp)
        localization_ids = track['association']['localizations']

        if len(localization_ids) > 2:
            print(f"Track {track['id']} is more than a pair")
            continue
        elif len(localization_ids) < 2:
            print(f"Track {track['id']} is less than a pair")
            continue

        if track_data.num_detection_images() == 2:
            print(f"Skipping already downloaded track {track_id}")
        else:
            # From downloading we put each track as to images
            code,detection_bgr_list = tator.StateGraphic.get_bgr(track_id)
            if code != 200:
                print("Failed to get track graphic!")
                continue

            for d_idx,detection_bgr in enumerate(detection_bgr_list):
                track_data.save_detection_image(detection_bgr, d_idx)
        # we always start at 0 for each track to the lookup in model_data.py can work
        index0.append(0+offset)
        index1.append(1+offset)
        if track['attributes']['Match'].find("Yes") >= 0:
            label.append(0)
        else:
            label.append(1)

        track_dirs_fp.write(os.path.join("tracks", f"{track_id:09d}")+'\n')
        offset += 2
    num_neg = np.sum(label)
    num_pos= np.size(label) - num_neg
    validate_stop = int(args.holdout_fraction * float(len(index0)))
    print("Number positive pairs: {}".format(num_pos))
    print("Number negative pairs: {}".format(num_neg))
    print("Saving pairing data...")
    model_data.save_cnn_validate(
        np.transpose([
            index0[:validate_stop],
            index1[:validate_stop]]),
        np.expand_dims(label[:validate_stop], axis=1))
    model_data.save_cnn_training(
        np.transpose([
            index0[validate_stop:],
            index1[validate_stop:]]),
        np.expand_dims(label[validate_stop:], axis=1))
