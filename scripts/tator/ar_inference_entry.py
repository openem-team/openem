#!/usr/bin/env python3

import json
import logging
import os
import subprocess
import shutil
import sys
from tarfile import TarFile
from time import sleep
import yaml

import docker


log_filename = "ar_inference_entry.log"


logging.basicConfig(
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


if __name__ == "__main__":
    try:
        config_file = os.getenv("CONFIG_FILE")
        work_dir = os.getenv("TATOR_WORK_DIR")

        if not os.path.isfile(config_file):
            raise ValueError(f"Could not find config file '{config_file}', exiting")

        with open(config_file, "r") as fp:
            config = yaml.safe_load(fp)

        state_type_id = config["state_type_id"]
        sample_size = config["sample_size"]
        attribute_name = config["attribute_name"]
        upload_version = config["upload_version"]
        video_order = config["video_order"]

        # Copy model files
        data_image = config["data_image"]
        client = docker.from_env()
        client.images.pull(data_image)
        container = client.containers.create(data_image)
        bits, _ = container.get_archive("/data")
        data_tar = os.path.join(work_dir, "data.tar")
        with open(data_tar, "wb") as fp:
            for chunk in bits:
                fp.write(chunk)

        with TarFile(data_tar) as fp:
            fp.extract("data/model_config.yaml", work_dir)
            fp.extract("data/deploy_model.pth", work_dir)

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
            work_dir,
            "--project-id",
            os.getenv("TATOR_PROJECT_ID"),
            "--attribute-name",
            attribute_name,
            "--upload-version",
            str(upload_version),
            "--multiview-ids",
            *os.getenv("TATOR_MEDIA_IDS").split(","),
            "--state-type",
            str(state_type_id),
            "--model-config-file",
            os.path.join(work_dir, "data", "model_config.yaml"),
            "--sample-size",
            str(sample_size),
            "--video-order",
            *[str(vid) for vid in video_order],
        ]

        cmd = " ".join(args)
        logger.info(f"Feature Extraction Command = '{cmd}'")
        p = subprocess.Popen(args)
        p.wait()
        sys.exit(p.returncode)
    except SystemExit:
        # sys.exit() raises a SystemExit exception, so just reraise it here for a happy exit
        raise
    except:
        # Some other exception was raised, sleep to keep the container alive for inspection
        logger.error("Failed with exception", exc_info=True)
        sleep(600)
