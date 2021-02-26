#!/usr/bin/env python3
import argparse
from datetime import datetime
from itertools import cycle
import json
import logging
from math import floor
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import List, Optional, Union
from uuid import uuid4

import boto3
import cv2
import numpy as np
from pandas import DataFrame
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import tator


if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.HalfTensor")


log_filename = "feature_extractor.log"


logging.basicConfig(
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


class ResNet50FeatureExtractor:
    """
    Extracts features from a video file.
    """

    class ResNet50Features(nn.Module):
        def __init__(self):
            super().__init__()
            image_modules = list(torchvision.models.resnet50(pretrained=True).children())[:-1]
            image_modules[-1] = nn.AdaptiveAvgPool2d(1)
            self.model = nn.Sequential(*image_modules)

        def forward(self, image):
            x = self.model(image)
            return x

    def __init__(
        self,
        frame_modulus: int,
        image_size: List[int],
        gpu_num: Optional[int] = 0,
        frame_cutoff: Optional[int] = float("inf"),
        frame_formatters: Optional[int] = 5,
        verbose: Optional[bool] = False,
    ):

        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_num)
            self._torch_device = torch.device(f"cuda:{gpu_num}")
        else:
            self._torch_device = torch.device("cpu")

        self._frame_cutoff = frame_cutoff
        self._frame_modulus = frame_modulus
        self._image_width = image_size[0]
        self._image_height = image_size[1]
        self._verbose = verbose

        self._raw_queue = mp.Queue(20)
        self._frame_queue = mp.Queue(20)
        self._stop_event = mp.Event()
        self._frame_stop_event = mp.Event()
        self._done_event = mp.Event()

        self._video_path = None
        self._read_frames_process = None
        self._enqueue_frames_processes = [None for _ in range(frame_formatters)]

    @staticmethod
    def _terminate_if_alive(process: Union[mp.Process, None]):
        if process and process.is_alive():
            process.terminate()

    def _start(self):
        # Clear out queues
        while not self._raw_queue.empty():
            self._raw_queue.get_nowait()
        while not self._frame_queue.empty():
            self._frame_queue.get_nowait()

        # Clear Events
        self._stop_event.clear()
        self._frame_stop_event.clear()
        self._done_event.clear()

        # Terminate old processes
        self._terminate_if_alive(self._read_frames_process)
        for process in self._enqueue_frames_processes:
            self._terminate_if_alive(process)

        # Start frame processors
        self._read_frames_process = mp.Process(target=self._read_frames)
        self._read_frames_process.daemon = True
        self._read_frames_process.start()

        for idx in range(len(self._enqueue_frames_processes)):
            p = mp.Process(target=self._enqueue_frames)
            p.daemon = True
            p.start()
            self._enqueue_frames_processes[idx] = p

    def _read_frames(self):
        if self._video_path is None:
            raise RuntimeError("Video path not given, cannot read frames")

        ok = True
        vid = cv2.VideoCapture(self._video_path)

        # TODO Unnecessary? Don't use vid_len or fps anywhere
        if cv2.__version__ >= "3.2.0":
            vid_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = vid.get(cv2.CAP_PROP_FPS)
        else:
            vid_len = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
            fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)

        frame_num = 0
        while ok and not self._stop_event.is_set():
            if not self._raw_queue.full():
                ok, img = vid.read()
                if ok and (frame_num % self._frame_modulus == 0):
                    self._raw_queue.put((img, frame_num))
                frame_num += 1
            if frame_num > self._frame_cutoff:
                break

        self._done_event.wait()
        self._stop_event.set()

    def _enqueue_frames(self):
        while not self._stop_event.is_set():
            try:
                img, frame_num = self._raw_queue.get(timeout=2)
                formatted_img = self._format_img(img)
                self._frame_queue.put((formatted_img, frame_num))
            except:
                self._done_event.wait()
                self._stop_event.set()

    def _format_img(self, img):
        start_time = time.time()
        if self._image_width > 0 and self._image_height > 0:
            img = cv2.resize(
                img, (self._image_width, self._image_height), interpolation=cv2.INTER_NEAREST
            )
        img = img[:, :, (2, 1, 0)]
        img = img.astype(np.float32)
        img /= float(255.0)
        img = img.transpose((2, 0, 1))
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        for i in range(3):
            img[i, :, :] -= float(img_mean[i])
            img[i, :, :] /= float(img_std[i])
        if self._verbose:
            logger.info(f"Image preprocessing time: {time.time() - start_time}")

        return img

    def _get_frames(self, n):
        timeout = 2  # seconds to wait for an item in the queue

        # The first .get() is outside the try..except to notify the caller that there are no frames
        # left to get
        result = [self._frame_queue.get(timeout=timeout)]
        while len(result) < n:
            # .get() more until `self._frame_queue` is empty or `n` items obtained
            try:
                result.append(self._frame_queue.get(timeout=timeout))
            except queue.Empty:
                break

        images, frames = map(list, zip(*result))
        images = np.array(images)
        return images, frames

    def extract_features(self, video_path: str) -> DataFrame:
        current_time = datetime.now().strftime("%H:%M:%S")
        logger.info(f"Starting processing at: {current_time}")
        self._video_path = video_path
        self._start()
        batch_size = 32

        with torch.no_grad():
            model = self.ResNet50Features().to(self._torch_device)
            model.eval()
            video_features = {}
            while True:
                st = time.time()
                try:
                    images, frames = self._get_frames(batch_size)
                except queue.Empty:
                    if self._verbose:
                        logger.info("timed out getting frames")
                    self._done_event.set()
                    break

                bt = time.time()
                images = torch.Tensor(images, device=self._torch_device)
                ppt = time.time()

                if self._verbose:
                    logger.info(f"Elapsed time pre-process = {ppt - st}")
                model_out = model(images)
                mt = time.time()
                if self._verbose:
                    logger.info(f"Elapsed time model = {mt - st}")
                image_features = model_out.data.cpu().numpy()
                orig_shape = image_features.shape
                image_features = np.squeeze(image_features)
                if orig_shape[0] == 1:
                    image_features = np.expand_dims(image_features, 0)
                for frame_num, features in zip(frames, image_features):
                    video_features[frame_num] = features
                at = time.time()

                if self._verbose:
                    logger.info(f"\tget frames time\t\t\t= {bt - st}")
                    logger.info(f"\timages to gpu time\t\t= {ppt - bt}")
                    logger.info(f"\tmodel time\t\t\t= {mt - ppt}")
                    logger.info(f"\taggregate features time\t\t= {at - mt}")
                    logger.info(f"\tapproximate processing fps\t= {batch_size/(at - st)}")

            video_features_df = DataFrame.from_dict(video_features, orient="index")
            video_features_df.sort_index(inplace=True)
            self._frame_stop_event.set()
            self._stop_event.set()

        current_time = datetime.now().strftime("%H:%M:%S")
        logger.info(f"Finished processing at: {current_time}")

        return video_features_df


def _any_remaining(media_tracker, key):
    """ Returns True if at least one value for the given key is None """
    return next((True for v in media_tracker.values() if v[key] is None), False)


def main(
    media_ids,
    project_id,
    force_extraction,
    attribute_name,
    api,
    work_dir,
    frame_modulus,
    image_size,
    verbose,
    s3_bucket,
    endpoint_url,
    access_key,
    secret_key,
):
    # Download media
    logger.info("Downloading media")
    media_elements = api.get_media_list(project_id, media_id=media_ids)

    # Create dict for tracking the progress of each media item
    media_tracker = {}
    for element in media_elements:
        s3_info = element.attributes[attribute_name]
        s3_key = None
        if s3_info and not force_extraction:
            try:
                s3_key = json.loads(s3_info)["key"]
            except:
                pass

            if s3_key:
                logger.info(f"Features exist for {element.name}, skipping extraction")
                continue

        if element.media_files.streaming is None:
            logger.info(f"No video file found for {element.name}, skipping extraction")
            continue

        logger.info(f"Scheduling {element.name} for feature extraction")
        media_tracker[element.id] = {
            "element": element,
            "media_file": None,
            "download_attempts_rem": 5,
            "df_file": None,
            "s3_key": None,
        }

    while _any_remaining(media_tracker, "media_file"):
        for media_dict in media_tracker.values():
            # Check to see if the file has already been downloaded or if the features are already in
            # s3
            if media_dict["media_file"] is not None:
                continue

            media = media_dict["element"]
            media_unique_name = f"{media.id}_{media.name}"
            media_filepath = os.path.join(work_dir, media_unique_name)
            try:
                for _ in tator.download_media(api, media, media_filepath):
                    pass
            except:
                logger.warning(f"{media_unique_name} did not download!")
                media_dict["download_attempts_rem"] -= 1
                continue

            if not os.path.exists(media_filepath):
                media_dict["download_attempts_rem"] -= 1
                logger.warning(f"{media_unique_name} did not download!")
            else:
                media_dict["media_file"] = media_filepath
        for media_id in list(media_tracker.keys()):
            if media_tracker[media_id]["download_attempts_rem"] < 1:
                media_tracker.pop(media_id)

    if len(media_tracker) == 0:
        logger.warning("No media requiring extraction")
        return

    logger.info("Media successfully downloaded")

    logger.info("Extracting features")
    df_files = []
    rfe = ResNet50FeatureExtractor(
        frame_modulus=frame_modulus, image_size=image_size, verbose=verbose
    )
    while _any_remaining(media_tracker, "df_file"):
        for media_dict in media_tracker.values():
            # Check to see if the features have already been extracted or if the features are
            # already in s3
            if media_dict["df_file"] is not None:
                continue

            media_file = media_dict["media_file"]
            base_filename = os.path.splitext(media_file)[0]
            df_path = f"{base_filename}.hdf"

            # Extract features
            try:
                df = rfe.extract_features(media_file)
                df.to_hdf(df_path, mode="w", key="df", format="table")
                del df
            except:
                pass

            if not os.path.exists(df_path):
                logger.warning(f"Features not extracted for {media_file}!")
            else:
                media_dict["df_file"] = df_path

    logger.info("Features successfully extracted")

    # Push features to s3 and add feature location to media

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    logger.info("Uploading features to s3")
    while _any_remaining(media_tracker, "s3_key"):
        for media_id, media_dict in media_tracker.items():
            # Check to see if the features are already in s3
            if media_dict["s3_key"] is not None:
                continue

            uuid_filename = f"{uuid4()}.hdf"
            try:
                client.upload_file(media_dict["df_file"], s3_bucket, uuid_filename)
            except:
                logger.warn(f"Unable to upload file {uuid_filename} to s3")
                continue

            try:
                feature_s3 = f'{{"bucket": "{s3_bucket}", "key": "{uuid_filename}"}}'
                api.update_media(int(media_id), {"attributes": {attribute_name: feature_s3}})
            except:
                logger.warn(f"Unable to set {attribute_name} attribute")
                continue

            media_dict["s3_key"] = uuid_filename

    logger.info("Features uploaded to s3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tator.get_parser(parser)
    parser.add_argument("--access-key", type=str, required=True)
    parser.add_argument("--secret-key", type=str, required=True)
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--endpoint-url", type=str, required=True)
    parser.add_argument("--attribute-name", type=str, required=True)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--media-ids", type=int, nargs="*", required=True)
    parser.add_argument("--frame-modulus", type=int, required=True)
    parser.add_argument("--image-size", type=int, nargs=2, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force-extraction", action="store_true")
    args = parser.parse_args()

    logger.info(f"ARGS: {args}")
    logger.info(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for feature extraction")

    main(
        media_ids=args.media_ids,
        project_id=args.project_id,
        force_extraction=args.force_extraction,
        attribute_name=args.attribute_name,
        api=tator.get_api(args.host, args.token),
        work_dir=args.work_dir,
        frame_modulus=args.frame_modulus,
        image_size=args.image_size,
        verbose=args.verbose,
        s3_bucket=args.s3_bucket,
        endpoint_url=args.endpoint_url,
        access_key=args.access_key,
        secret_key=args.secret_key,
    )
