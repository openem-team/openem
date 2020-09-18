#!/usr/bin/env python3

""" Script to upload detection pairs to tator. """

import argparse
import logging
import math
import sys

import progressbar
import tator
import pandas as pd

# pylint: disable=C0103
# pylint: disable=W1203

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)

count = 0
def upload_pair(pair, progress_bar, results):
    global count
    count += 1
    progress_bar.update(count)
    match = "Unknown"

    # Faster to fetch by media and cache it
    media_id = int(pair['media'])
    if media_id in tracks_by_media:
        tracks = tracks_by_media[media_id]
    else:
        tracks_by_media[media_id] = api.get_state_list(args.project, media_id=[media_id], type=args.tracklet_type_id)
        tracks = tracks_by_media[media_id]
        
    for track in tracks:
        if track.id == int(pair['first']):
            track_l = track
        elif track.id == int(pair['second']):
            track_r = track
    obj = {"localization_ids" : [*track_l.localizations,
                                 *track_r.localizations],
           "type": args.state_type_id,
           "media_ids" : [int(pair['media'])],
           "Frame Delta" : int(pair.frame_delta),
           "Tracklet Match": match,
           "modified": True
    }
    if args.version_id:
        obj.update({"version": args.version_id})
    results.append(obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = tator.get_parser(parser)
    parser.add_argument("--state-type-id",
                        help="ID to use for tracklet pair ingestion",
                        type=int,
                        required=True)
    parser.add_argument("--tracklet-type-id",
                        help="ID of underlying tracklets",
                        type=int,
                        required=True)
    parser.add_argument("--version-id",
                        help="ID for the version to upload",
                        type=int,
                        required=True)
    parser.add_argument("--project",
                        help="ID for the project",
                        type=int,
                        required=True)
    parser.add_argument("pairs_csv",
                        help="output file")

    args = parser.parse_args()

    pairs_df = pd.read_csv(args.pairs_csv)

    api = tator.get_api(args.host, args.token)

    pBar = progressbar.ProgressBar(redirect_stdout=True,
                                   redirect_stderr=True,
                                   max_value=len(pairs_df))
    pBar.update(0)
    results=list()
    tracks_by_media={}
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
        api.create_state_list(args.project, results[start:end])
