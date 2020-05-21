#!/usr/bin/env python3

import argparse
import openem
import os
import cv2
import numpy as np
from openem.tracking import FeaturesComparator
import json
import sys
import datetime
import pytator
from pprint import pprint

def _intersection_over_union(boxA, boxB):
    """ Computes intersection over union for two bounding boxes.
        Inputs:
        boxA -- First box. Must be a dict containing x, y, width, height.
        boxB -- Second box. Must be a dict containing x, y, width, height.
        Return:
        Intersection over union.
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(int(boxA["x"]), int(boxB["x"]))
    yA = max(int(boxA["y"]), int(boxB["y"]))
    xB = min(int(boxA["x"]) + int(boxA["width"]),
             int(boxB["x"]) + int(boxB["width"]))
    yB = min(int(boxA["y"]) + int(boxA["height"]),
             int(boxB["y"]) + int(boxB["height"]))

    # compute the area of intersection rectangle
    interX = xB - xA + 1
    interY = yB - yA + 1
    if interX < 0 or interY < 0:
        iou = 0.0
    else:
        interArea = float((xB - xA + 1) * (yB - yA + 1))
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = int(boxA["width"]) * int(boxA["height"])
        boxBArea = int(boxB["width"]) * int(boxB["height"])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        if float(boxAArea + boxBArea - interArea) <= 0.0:
            return 0.00
        try:
            iou = interArea / float(boxAArea + boxBArea - interArea)
        except Exception as e:
            print(e)
            print("interArea: {}".format(interArea))
            print("Union: {}".format(float(boxAArea + boxBArea - interArea)))
        # return the intersection over union value
    return iou

def crop_localization(frame_bgr, localization):
    img_width = frame_bgr.shape[1]
    img_height = frame_bgr.shape[0]
    box_x = round(localization['x'] * img_width)
    box_y = round(localization['y'] * img_height)
    box_width = round(localization['width'] * img_width)
    box_height = round(localization['height'] * img_height)
    img_crop = frame_bgr[box_y:box_y+box_height,box_x:box_x+box_width,:]
    return img_crop

def make_box(shape, det):
    return {'x': det['x'] * shape[1],
            'y': det['y'] * shape[0],
            'width': det['width'] * shape[1],
            'height': det['height'] * shape[0]}
def find_pairs(media_shape,frameDets, nextDets):
    pairs = []
    for det in frameDets:
        for fut in nextDets:
            iou = _intersection_over_union(make_box(media_shape,det),
                                           make_box(media_shape,fut))
            if iou <= 0:
                print("Skipping detections with no overlap")
                continue
            comparator.addPair(det['bgr'], fut['bgr'])
            label = comparator.process()
            # Threshold this?
            if round(label[0][0]) == 0:
                pairs.append([det['id'], fut['id']])

    if len(pairs) > 0:
        return pairs
    else:
        return None

def process_detections(media_id, media_shape, localizations_by_frame):
    frames = localizations_by_frame.keys()
    pairs_by_frame={}
    for frame in frames:
        if frame+1 in localizations_by_frame:
            frameDets = localizations_by_frame[frame]
            nextDets = localizations_by_frame[frame+1]
            pairs = find_pairs(media_shape, frameDets, nextDets)
            if pairs:
                pairs_by_frame[frame] = pairs

    return pairs_by_frame

def link_tracklets(pairs_by_frame):
    tracklets=[]
    # Iterate over the pairs and link into contiguous chains
    for idx,(frame, pairs) in enumerate(pairs_by_frame.items()):
        for pair in pairs:
            match = False
            for tracklet in tracklets:
                if tracklet[-1] == pair[0]:
                    match = True
                    tracklet.append(pair[1])
            if match is False:
                tracklets.append(pair)
    return tracklets


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    pytator.tator.cli_parser(parser)
    parser.add_argument("--model-file", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--detection-type-id", type=int, required=True)
    parser.add_argument("--tracklet-type-id", type=int, required=True)
    parser.add_argument('media_files', type=str, nargs='+')
    args = parser.parse_args()

    tator = pytator.Tator(args.url, args.token, args.project)

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

        vid=cv2.VideoCapture(media_file)
        ok=True
        frame = 0
        media_shape = None
        while ok:
            ok,frame_bgr = vid.read()
            if media_shape is None:
                media_shape = frame_bgr.shape

            if frame in localizations_by_frame:
                for l in localizations_by_frame[frame]:
                    l['bgr'] = crop_localization(frame_bgr, l)
            frame+=1

        # Generate localization bgr based on grouped localizations
        pairs = process_detections(media_id, media_shape, localizations_by_frame)
        print(f"{len(pairs)} Pairs found")
        tracklets = link_tracklets(pairs)
        print(f"{len(tracklets)} Tracklets found")
        pprint(tracklets)
        for tracklet in tracklets:
            tator.Track.new({"type": args.tracklet_type_id,
                             "media_ids": [int(media_id)],
                             "localization_ids": tracklet,
                             "Species": "Tracklet"})
