#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import shutil
import sys


log_filename = "feature_extraction_entry.log"


logging.basicConfig(
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # TODO remove pipeline_args_str if unneeded
    pipeline_args_str = os.getenv("TATOR_PIPELINE_ARGS")
    if pipeline_args_str:
        pipeline_args = json.loads(pipeline_args_str)
    else:
        pipeline_args = {}

    args = [
        "python3",
        "/scripts/feature_extractor.py",
        "--host",
        os.getenv("TATOR_API_SERVICE").replace("/rest", ""),
        "--token",
        os.getenv("TATOR_AUTH_TOKEN"),
        "--access-key",
        os.getenv("OBJECT_STORAGE_ACCESS_KEY"),
        "--secret-key",
        os.getenv("OBJECT_STORAGE_SECRET_KEY"),
        "--s3-bucket",
        os.getenv("S3_BUCKET"),
        "--endpoint-url",
        os.getenv("ENDPOINT_URL"),
        "--work-dir",
        os.getenv("TATOR_WORK_DIR"),
        "--project-id",
        os.getenv("TATOR_PROJECT_ID"),
        "--attribute-name",
        os.getenv("TATOR_ATTRIBUTE_NAME"),
        "--media-ids",
        *os.getenv("TATOR_MEDIA_IDS").split(","),
        "--frame-modulus",
        os.getenv("FRAME_MODULUS"),
        "--image-size",
        *os.getenv("IMAGE_SIZE").split(","),
    ]

    if os.getenv("VERBOSE") is not None:
        args.append("--verbose")

    if os.getenv("FORCE_EXTRACTION") is not None:
        args.append("--force-extraction")

    cmd = " ".join(args)
    logger.info(f"Feature Extraction Command = '{cmd}'")
    p = subprocess.Popen(args)
    p.wait()
    sys.exit(p.returncode)
