#!/usr/bin/env python3

""" Script to upload detection pairs to tator. """

import argparse
import logging
import sys

import progressbar
import pytator
import pandas as pd

# pylint: disable=C0103
# pylint: disable=W1203

logger = logging.getLogger("upload_pairs.py")
logging.basicConfig(level=logging.INFO)

count = 0
def upload_pair(pair, progress_bar):
    global count
    count += 1
    progress_bar.update(count)
    obj = {"localization_ids" : [pair['first'], pair['second']],
           "Species" : pair.species,
           "IoU" : float(pair.iou),
           "Match": "Unknown",
           "type": args.state_type_id,
           "media_ids" : pair['media']}
    tator.Track.new(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = pytator.tator.cli_parser(parser)
    parser.add_argument("--state-type-id",
                        help="ID to use for pair ingestion")
    parser.add_argument("pairs_csv",
                        help="output file")

    args = parser.parse_args()

    pairs_df = pd.read_csv(args.pairs_csv)
    
    tator = pytator.Tator(args.url, args.token, args.project)
    
    pBar = progressbar.ProgressBar(redirect_stdout=True,
                                   redirect_stderr=True,
                                   max_value=len(pairs_df))
    pBar.update(0)
    pairs_df.apply(upload_pair, axis=1, args=(pBar,))
    pBar.finish()

    
