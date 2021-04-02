#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import shutil
import sys


log_filename = "ar_inference_entry.log"


logging.basicConfig(
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    args = [
        "python3",
        "/scripts/ar_inference.py",
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
        "--multiview-ids",
        *os.getenv("TATOR_MEDIA_IDS").split(","),
        "--state-type",
        os.getenv("STATE_TYPE"),
        "--model-config-file",
        os.getenv("MODEL_CONFIG_FILE"),
        "--sample-size",
        os.getenv("SAMPLE_SIZE"),
        "--video-order",
        *os.getenv("VIDEO_ORDER").split(","),
    ]

    if os.getenv("FORCE_EXTRACTION") is not None:
        args.append("--force-extraction")

    cmd = " ".join(args)
    logger.info(f"Feature Extraction Command = '{cmd}'")
    p = subprocess.Popen(args)
    p.wait()
    sys.exit(p.returncode)
