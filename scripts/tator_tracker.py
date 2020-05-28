#!/usr/bin/env python3

import argparse
import openem
import os
import cv2
import numpy as np
from openem.tracking import *
import json
import sys
import datetime
import pytator
from pprint import pprint
from collections import defaultdict

def crop_localization(frame_bgr, localization):
    img_width = frame_bgr.shape[1]
    img_height = frame_bgr.shape[0]
    box_x = round(localization['x'] * img_width)
    box_y = round(localization['y'] * img_height)
    box_width = round(localization['width'] * img_width)
    box_height = round(localization['height'] * img_height)
    img_crop = frame_bgr[box_y:box_y+box_height,box_x:box_x+box_width,:]
    return img_crop

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    pytator.tator.cli_parser(parser)
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--detection-type-id", type=int, required=True)
    parser.add_argument("--tracklet-type-id", type=int, required=True)
    parser.add_argument("--version-number", type=int)
    parser.add_argument('media_files', type=str, nargs='+')
    args = parser.parse_args()

    tator = pytator.Tator(args.url, args.token, args.project)
    version_id = None
    if args.version_number:
        for version in tator.Version.all():
            if version['number'] == args.version_number:
                version_id = version['id']
                print(f"Using version ID {version_id}")

    #extractor=FeaturesExtractor(args.model_file)
    comparator=FeaturesComparator(args.model_file)

    localizations_by_frame = {}
    for media_file in args.media_files:
        comps=os.path.splitext(os.path.basename(media_file))[0]
        media_id=comps.split('_')[0]
        lookup = {"type": args.detection_type_id,
                  "media_id" : media_id}
        localizations = tator.Localization.filter(lookup)
        print(f"Processing {len(localizations)} detections")
        # Group by localizations by frame
        for lid, local in enumerate(localizations):
            frame = local['frame']
            if frame in localizations_by_frame:
                localizations_by_frame[frame].append(local)
            else:
                localizations_by_frame[frame] = [local]

        # TODO extract appearance here instead
        vid=cv2.VideoCapture(media_file)
        ok=True
        frame = 0
        media_shape = None
        detections=[]
        track_ids=[]
        track_id=1
        while ok:
            ok,frame_bgr = vid.read()
            if media_shape is None:
                media_shape = frame_bgr.shape

            if frame in localizations_by_frame:
                for l in localizations_by_frame[frame]:
                    l['bgr'] = crop_localization(frame_bgr, l)
                    detections.append(l)
                    track_ids.append(track_id)
                    track_id += 1
            frame+=1

        track_ids = renumber_track_ids(track_ids)
        # Generate localization bgr based on grouped localizations
        detections, track_ids, pairs, weights, is_cut, constraints = join_tracklets(
            detections,
            track_ids,
            1,
            comparator,
            None,
            None,
            media_shape[1],
            media_shape[0],
            15,
            0.0,
            args.batch_size)

        # Now we make new track objects based on the result
        # from the graph solver
        # [ detection, detection, detection, ...]
        # [ track#, track#, track#,...]
        # [ 133, 33, 13, 133,]
        # [ 0,0,1,1]
        # TODO: Handle is_cut?
        def join_up_final(detections, track_ids):
            tracklets = defaultdict(list)
            num_tracklets = np.max(track_ids) + 1
            assert(len(detections) == len(track_ids))
            for d,tid in zip(detections, track_ids):
                tracklets[tid].append(d['id'])
            return tracklets
        tracklets = join_up_final(detections, track_ids)
        new_objs=[{"type": args.tracklet_type_id,
                   "media_ids": [int(media_id)],
                   "localization_ids": tracklet,
                   "Species": "Tracklet",
                   "version": version_id} for tracklet in tracklets.values()]
        tator.Track.new(new_objs)
