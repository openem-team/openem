#!/usr/bin/env python3

import argparse
import datetime
from itertools import cycle
import logging
from math import floor
import os
import sys
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import tator


log_filename = "feature_extractor.log"


logging.basicConfig(
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


# Dummy feature extractor that fills a 1x2048 array with the frame number for each frame
def extract_features(vid):
    frame_num = 0
    ok, _ = vid.read()

    feature_list = []
    while ok:
        feature_list.append([frame_num] * 2048)
        ok, _ = vid.read()
        frame_num += 1

    return pd.DataFrame(feature_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tator.get_parser(parser)
    # Access/secret keys and bucket only needed if this is not a dry-run
    parser.add_argument("--access-key", required="--dry-run" not in sys.argv, type=str)
    parser.add_argument("--secret-key", required="--dry-run" not in sys.argv, type=str)
    parser.add_argument("--s3-bucket", required="--dry-run" not in sys.argv, type=str)
    parser.add_argument("--endpoint-url", required="--dry-run" not in sys.argv, type=str)
    parser.add_argument("--attribute-name", required="--dry-run" not in sys.argv, type=str)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--media-ids", type=int, nargs="*")
    args = parser.parse_args()

    media_ids = args.media_ids
    project_id = args.project_id
    api = tator.get_api(args.host, args.token)

    # Download media
    logger.info("Downloading media")
    media_elements = api.get_media_list(project_id, media_id=media_ids)
    n_media = len(media_ids)
    media_files_to_process = []
    for (idx, (media_id, media)) in enumerate(zip(media_ids, media_elements)):
        media_unique_name = f"{media.id}_{media.name}"
        media_filepath = os.path.join(args.work_dir, media_unique_name)
        for _ in tator.download_media(api, media, media_filepath):
            pass
        if not os.path.exists(media_filepath):
            print("File did not download!")
            sys.exit(255)

        media_files_to_process.append(media_filepath)

    logger.info("Media successfully downloaded")

    logger.info("Extracting features")
    n_files = len(media_files_to_process)
    df_files = []
    for idx, media_file in enumerate(media_files_to_process):
        base_filename = os.path.splitext(media_file)[0]
        df_path = f"{base_filename}.hdf"
        if os.path.isfile(df_path):
            logger.info(f"Feature file found for {media_file}, skipping extraction")
        else:
            # Extract features
            df = extract_features(cv2.VideoCapture(media_file))
            df.to_hdf(df_path, key=base_filename, mode="w")
        df_files.append(df_path)

    logger.info("Features extracted")

    # Push features to s3 and add feature location to media
    if not args.dry_run:
        access_key = args.access_key
        secret_key = args.secret_key
        s3_bucket = args.s3_bucket
        endpoint_url = args.endpoint_url
        attribute_name = args.attribute_name

        if not all(
            bool(arg) for arg in [access_key, secret_key, s3_bucket, endpoint_url, attribute_name]
        ):
            raise RuntimeError(
                f"Must set all of the following arguments: --access-key (received '{access_key}'),"
                f" --secret-key (received '{secret_key}'), --s3-bucket (received '{s3_bucket}'),"
                f" --endpoint-url (received {endpoint_url}),"
                f" --attribute-name (received '{attribute_name}')."
            )
        import boto3

        client = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )

        logger.info("Uploading features to s3")
        for idx, df_file in enumerate(df_files):
            uuid_filename = f"{uuid4()}.hdf"
            try:
                client.upload_file(df_file, s3_bucket, uuid_filename)
            except:
                logger.warn(f"Unable to upload file {uuid_filename} to s3")
                continue

            try:
                feature_s3 = f'{{"bucket": "{s3_bucket}", "key": "{uuid_filename}"}}'
                api.update_media(int(media_id), {"attributes": {attribute_name: feature_s3}})
            except:
                logger.warn(f"Unable to set {attribute_name} attribute")

        logger.info("Features uploaded to s3")

    else:
        logger.info("Dry run, files not being uploaded to s3")
