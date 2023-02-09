from textwrap import dedent
from urllib.parse import urlparse
import argparse
import datetime
import os
import sys
import traceback

import pandas as pd
import tator
from tator.util._upload_file import _upload_file

import visual_iou_tracker

def parse_args() -> argparse.Namespace:
    """ Process script arguments
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--url", type=str, help="URL to Tator REST service.")
    parser.add_argument("--token", type=str, help="Tator API token associated with user who esecuted the workflow")
    parser.add_argument("--media-ids", type=str, help="Media ID of section to process. Multiple IDs are expected to be comma separated.")
    parser.add_argument('--max-coast-age', type=int, help='Maximum track coast age', default=5)
    parser.add_argument('--association-threshold', type=float, help='Passing association threshold', default=0.1)
    parser.add_argument('--min-num-detections', type=int, help='Minimum number of detections (not generated by tracker) for a valid track', default=1)
    parser.add_argument('--min-total-confidence', type=float, help='Minimum total sum of detection confidences for a valid track', default=-1.0)
    parser.add_argument('--track-type', type=int)
    parser.add_argument('--detection-type', type=int)
    parser.add_argument('--detection-version', type=int)
    parser.add_argument('--track-version', type=int)
    parser.add_argument('--extend-track', action="store_true")
    parser.add_argument("--start-frame", type=int)
    return parser.parse_args()


if __name__ == '__main__':

    # Parse arguments and set up API
    args = parse_args()

    print("Arguments: ")
    print(f"media(s): {args.media_ids}")

    url = urlparse(args.url)
    host = f"{url.scheme}://{url.netloc}"
    tator_api = tator.get_api(host=host, token=args.token)

    tokens = args.media_ids.split(",")
    media_ids = [int(token) for token in tokens]

    for media_id in media_ids:
      visual_iou_tracker.process_media(
          tator_api=tator_api,
          media_id=media_id,
          local_video_file_path="",
          max_coast_age=args.max_coast_age,
          association_threshold=args.association_threshold,
          min_num_detections=args.min_num_detections,
          min_total_confidence=args.min_total_confidence,
          detection_type_id=args.detection_type,
          state_type_id=args.track_type,
          detection_version=args.detection_version,
          track_version=args.track_version,
          extend_track=args.extend_track,
          start_frame=args.start_frame)
