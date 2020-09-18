#!/usr/bin/env python3

""" Script to upload detection pairs to tator. """

import argparse
import logging
import math
import sys

import progressbar
import pytator
import pandas as pd

# pylint: disable=C0103
# pylint: disable=W1203

logger = logging.getLogger("upload_pairs.py")
logging.basicConfig(level=logging.INFO)

count = 0
def upload_pair(pair, progress_bar, results):
    global count
    count += 1
    progress_bar.update(count)
    iou = float(pair.iou)
    match = "Unknown"
    if args.iou_match and iou > args.iou_match:
        match = "Yes (Whole)"
    obj = {"localization_ids" : [pair['first'], pair['second']],
           "type": args.state_type_id,
           "media_ids" : [pair['media']],
           "Species" : pair.species,
           "IoU" : iou,
           "Match": match,
           "modified": True
    }
    if args.version_id:
        obj.update({"version": args.version_id})
    results.append(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = pytator.tator.cli_parser(parser)
    parser.add_argument("--state-type-id",
                        help="ID to use for pair ingestion",
                        type=int)
    parser.add_argument("--version-id",
                        help="ID for the version to upload",
                        type=int)
    parser.add_argument("--iou-match",
                        help="""If supplied will assume an IoU greater than this
                              is a match""",
                        type=float)
    parser.add_argument("pairs_csv",
                        help="output file")

    args = parser.parse_args()

    pairs_df = pd.read_csv(args.pairs_csv)

    tator = pytator.Tator(args.url, args.token, args.project)

    pBar = progressbar.ProgressBar(redirect_stdout=True,
                                   redirect_stderr=True,
                                   max_value=len(pairs_df))
    pBar.update(0)
    results=list()
    pairs_df.apply(upload_pair, axis=1, args=(pBar,results))
    pBar.finish()

    batch_size = 50
    batch_count = math.ceil(len(results) / batch_size)
    pBar = progressbar.ProgressBar(redirect_stdout=True,
                                   redirect_stderr=True,
                                   max_value=batch_count)
    for batch_idx in pBar(range(batch_count)):
        start = batch_idx * batch_size
        end = start + batch_size
        tator.Track.new(results[start:end])
