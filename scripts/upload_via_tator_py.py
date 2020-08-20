#!/usr/bin/env python3

""" Upload a result set or training set to tator for analysis """
import argparse
import csv
import progressbar
import sys
import signal
import os

import pandas as pd

import tator

def exit_func(_,__):
    print("SIGINT detected")
    os._exit(0)

def parse_args():
    """ Get the arguments passed into this script.

    Utilizes tator's parser which has its own based added arguments

    """
    parser = tator.get_parser()
    parser.add_argument("--csvfile", help="test.csv, length.csv, or detect.csv")
    parser.add_argument("--species-attr-name", type=str, default="Species")
    parser.add_argument("--confidence-attr-name", type=str, default="Confidence")
    return parser.parse_args()

def main():
    """ Main application thread
    """

    signal.signal(signal.SIGINT, exit_func)

    # Get those args
    args = parse_args()

    # Process the input data
    detect_keys = ['video_id', 'frame', 'x', 'y', 'w', 'h', 'det_conf', 'det_species']
    input_data = pd.read_csv(args.csvfile)
    if len(input_data) == 0:
        raise ValueError("Error: No input data")

    if detect_keys != list(input_data.keys()):
        raise ValueError("Error: Input data columns don't match expected values")

    # Connect to tator
    tator_api = tator.get_api(host=args.host, token=args.token)

    # There might be multiple media in the input data. Grab the different media
    # Loop through each of the unique media, and process each of the localizations
    unique_media = list(input_data.video_id.unique())
    for local_media_name in unique_media:

        print(f"Processing: {local_media_name}")

        # Extract the media ID from the name (expected format id_name)
        media_id = local_media_name.split('_')[0]

        # Grab the media information
        media = tator_api.get_media(id=media_id)

        # Grab the first localization type that is a box. It's assumed that this project
        # has been set up to only have one localization box type (that will be the detections)
        detection_type_id = None
        box_type_counts = 0
        localization_types = tator_api.get_localization_type_list(project=media.project)
        for loc_type in localization_types:
            if loc_type.dtype == 'box':
                detection_type_id = loc_type.id
                box_type_counts += 1

        if detection_type_id is None:
            raise ValueError("No localization box type detected. Expected only one.")

        if box_type_counts > 1:
            raise ValueError("Multiple localization box types detected. Expected only one.")

        # Grab the detections that match the current media name, loop through each of them
        # and create a localization
        det_df = input_data[input_data.video_id == local_media_name]
        print(f"...Number of detections to upload: {len(det_df)}")

        for idx, det in det_df.iterrows():

            x = 0.0 if det.x < 0 else det.x / media.width
            y = 0.0 if det.y < 0 else det.y / media.height

            width = det.w - det.x if det.x + det.w > media.width else det.w
            height = det.h - det.y if det.y + det.h > media.height else det.h
            width = width / media.width
            height = height / media.height

            attributes = {
                args.species_attr_name: det.det_species,
                args.confidence_attr_name: det.det_conf}

            detection_spec = dict(
                media_id=media.id,
                type=detection_type_id,
                frame=det.frame,
                x=x,
                y=y,
                width=width,
                height=height,
                **attributes)

            response = tator_api.create_localization_list(
                project=media.project,
                localization_spec=[detection_spec])

    print("fin.")

if __name__=="__main__":

    main()
