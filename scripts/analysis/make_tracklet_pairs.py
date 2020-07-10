#!/usr/bin/env python3

""" Script to generate tracklet pairs based on a configurable frame
    delta and IOU threshold

After generating one can do something like the following to generate a
sample of 10000 pairs to use as the upload_pairs.py script input

   >>> import pandas as pd
   >>> pairs = pd.read_csv('pairs.csv')
   >>> sampled=pairs.sample(n=10000)
   >>> sampled.to_csv('sampled.csv', index=False)

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

logger = logging.getLogger("make_tracklet_pairs.py")
logging.basicConfig(level=logging.WARNING)

from collections import defaultdict

def process_pairs(a_frame, b_frame):
    """ Given a set of boxes from 2 frames, return a list of matched pairs """
    pairs = []
    for a_box in a_frame:
        for b_box in b_frame:
            iou = intersection_over_union(a_box, b_box)
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
    tracklets = tator.State.filter({"type": args.tracklet_type_id,
                                "media_id": media_element['id']})
    if len(tracklets) == 0:
        return []
    logger.info(f"{media_element['name']} has {len(tracklets)} tracklets")
    group_by_end = defaultdict(list)
    group_by_start = defaultdict(list)
    pairs = []

    # Iterate over each box and group by frame
    for tracklet in tracklets:
        start_frame = tracklet['segments'][0][0]
        end_frame = tracklet['segments'][-1][-1]
        group_by_end[end_frame].append(tracklet)
        group_by_start[start_frame].append(tracklet)


    for end_frame,tracklet_ends in group_by_end.items():
        for tracklet_2 in tracklet_ends:
            for start_frame,tracklet_starts in group_by_start.items():
                for tracklet_1 in tracklet_starts:
                    frame_delta = start_frame - end_frame
                    if frame_delta > 0 and  frame_delta < args.max_frame_diff:
                        # Explode out localizations in each pair
                        pairs.append({"first": ":".join([str(x) for x in tracklet_1['localizations']]),
                                      "second": ":".join([str(x) for x in tracklet_2['localizations']]),
                                      "media": media_element['id'],
                                      "frame_delta": frame_delta})


    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = pytator.tator.cli_parser(parser)
    parser.add_argument("--section", help="Section name to process")
    parser.add_argument("--media-id", help="Individual media")
    parser.add_argument("--max-frame-diff", help="Only generate pairs with a maximum frame delta",
                        type=int,
                        default=32)
    parser.add_argument("--tracklet-type-id",
                        help="Tracklet Type ID from database",
                        required=True)
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

    df = pd.DataFrame(columns=["first", "second", "frame_delta", "media"])
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
                                           "frame_delta",
                                           "media"],
                                  data=pairs_in_media)
                df.to_csv(args.output_csv,
                          index=False,
                          header=False,
                          mode='a')
    logger.info(f"Finished")
