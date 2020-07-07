#!/usr/bin/env python3

""" Script to generate detection pairs based on a configurable frame
    delta and IOU threshold

After generating one can do something like the following to generate a
sample of 10000 pairs to use as the upload_pairs.py script input

   >>> import pandas as pd
   >>> pairs = pd.read_csv('pairs.csv')
   >>> sampled=pairs.sample(n=10000)
   >>> sampled.to_csv('sampled.csv', index=False)

Care should be taken to make sure a sampled set is a blend of positives and
negatives. IoU of a pair may be able to be used to determine this:

   >>> likely_not_matches = pairs.loc[pairs.iou < 0.50]

"""

import argparse
import logging
import time
import sys

import progressbar
import pytator
import pandas as pd

import multiprocessing
# pylint: disable=C0103
# pylint: disable=W1203

logger = logging.getLogger("make_detection_pairs.py")
logging.basicConfig(level=logging.WARNING)

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

def process_pairs(a_frame, b_frame):
    """ Given a set of boxes from 2 frames, return a list of matched pairs """
    pairs = []
    for a_box in a_frame:
        for b_box in b_frame:
            iou = _intersection_over_union(a_box, b_box)
            if a_box['attributes']['Species'] != b_box['attributes']['Species']:
                logger.warning("Species of boxes don't align")
                continue

            if iou > args.iou_threshold:
                pair = {"frame": a_box['frame'],
                        "first": a_box['id'],
                        "second": b_box['id'],
                        "iou": iou,
                        "species" : a_box['attributes']['Species']}
                pairs.append(pair)
            else:
                #print(f"iou 0 of {a_box} and {b_box}")
                pass

    return pairs
def process_media_file(media_element):
    """ Process a media element returning all the box pairs """
    boxes = tator.Localization.filter({"type": args.box_type_id,
                                       "media_id": media_element['id']})
    if boxes == None:
        return []
    logger.info(f"{media_element['name']} has {len(boxes)} boxes")
    group_by_frame = {}
    frames = set()
    pairs = []

    media_detail = tator.Media.get(media_element['id'])
    width = media_detail['width']
    height = media_detail['height']

    # Iterate over each box and group by frame
    for box in boxes:
        frame = box['frame']
        frames.add(frame)
        box['x'] *= width
        box['y'] *= height
        box['width'] *= width
        box['height'] *= height
        if frame in group_by_frame:
            # Convert back to natural coordinates
            group_by_frame[frame].append(box)
        else:
            group_by_frame[frame] = [box]

    # Create a vector of all the frames
    frames = list(frames)
    frames.sort()
    for frame in frames:
        next_frame = frame + args.frame_diff
        if next_frame not in group_by_frame:
            logger.info(f"{next_frame} has no data, so no pairs to match")
            continue
        media_pairs = process_pairs(group_by_frame[frame],
                                    group_by_frame[next_frame])
        for pair in media_pairs:
            pair.update({"media": media_element['id']})
        pairs.extend(media_pairs)

    # If a state type was supplied rectify the species
    if args.state_type_id:
        states = tator.State.filter({"type": args.state_type_id,
                                     "media_id": media_element['id']})
        if states is None:
            # This excludes media w/o state types to clarify species.
            logger.error(f"No matching states for {media_element['name']}")
            pairs=[]
            state_by_frame = {}
        else:
            state_by_frame = {}
            for state in states:
                frame = state['association']['frame']
                state_by_frame[frame] = state

        frames = state_by_frame.keys()
        frames = list(frames)
        for pair in pairs:
            closest = frames[0]
            for frame in frames:
                if abs(pair['frame']-frame) < abs(pair['frame']-closest):
                    closest = frame
            species = state_by_frame[closest]['attributes']['Species']
            pair['species'] = species


    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = pytator.tator.cli_parser(parser)
    parser.add_argument("--section", help="Section name to process")
    parser.add_argument("--media-id", help="Individual media")
    parser.add_argument("--frame-diff",
                        type=int,
                        help="Frame Delta",
                        default=1)
    parser.add_argument("--iou-threshold",
                        help="IOU threshold",
                        type=float,
                        default=0.0)
    parser.add_argument("--box-type-id",
                        help="Box Type ID from database",
                        required=True)
    parser.add_argument("--state-type-id",
                        help="ID to use for species association")
    parser.add_argument("--parallel-jobs",
                        type=int,
                        default=4)
    parser.add_argument("output_csv",
                        help="output file")

    args = parser.parse_args()

    if args.section and args.media_id:
        print("ERROR: Can't supply both section and media id")
        sys.exit(-1)
    elif args.section is None and args.media_id is None:
        print("ERROR: Must supply either --media-id or --section")
        sys.exit(-1)

    # Iterate over each media in section and get a list of
    # pairs
    tator = pytator.Tator(args.url, args.token, args.project)

    if args.section:
        medias = tator.Media.filter({"attribute":
                                     f"tator_user_sections::{args.section}"})
    else:
        medias = [tator.Media.get(args.media_id)]

    pBar = progressbar.ProgressBar(redirect_stdout=True,
                                   redirect_stderr=True)

    df = pd.DataFrame(columns=["first", "second", "frame", "iou", "species", "media"])
    df.to_csv(args.output_csv, index=False)

    pool = multiprocessing.Pool(processes=args.parallel_jobs)
    pending = []
    for media in pBar(medias):
        logger.info(f"Processing {media['name']}")
        job = pool.apply_async(process_media_file,(media,))
        pending.append(job)
        while len(pending) == args.parallel_jobs:
            for idx,job in enumerate(pending):
                if job.ready():
                    pairs_in_media = job.get()
                    del pending[idx]
                    df = pd.DataFrame(columns=["first",
                                               "second",
                                               "frame",
                                               "iou",
                                               "species",
                                               "media"],
                                      data=pairs_in_media)
                    df.to_csv(args.output_csv,
                              index=False,
                              header=False,
                              mode='a')
            time.sleep(0.250)

    # wait for any pending jobs left over
    while len(pending) > 0:
        for idx,job in enumerate(pending):
            if job.ready():
                pairs_in_media = job.get()
                del pending[idx]
                df = pd.DataFrame(columns=["first",
                                           "second",
                                           "frame",
                                           "iou",
                                           "species",
                                           "media"],
                                  data=pairs_in_media)
                df.to_csv(args.output_csv,
                          index=False,
                          header=False,
                          mode='a')
    logger.info(f"Finished")
