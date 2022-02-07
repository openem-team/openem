import argparse
import json
import logging
from math import isnan
import os
import pandas as pd
from statistics import median
from typing import Dict, Generator, List, Optional, Tuple
import yaml

import boto3
import torch
from torch import nn
import numpy as np

import tator


MAX_CHUNKED_CREATE_RETRIES = 5


if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


class ModelManager:
    class LSTMAR(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            batch_size: int,
            output_dim: int,
            num_layers: int,
            is_cuda: bool,
            dropout: float = 0,
        ):

            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.batch_size = batch_size
            self.num_layers = num_layers
            self._is_cuda = is_cuda
            self.dropout = dropout
            self.lstm = nn.LSTM(
                self.input_dim,
                self.hidden_dim,
                self.num_layers,
                dropout=self.dropout,
                bidirectional=True,
            )
            self.linear1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2)
            self.linear2 = nn.Linear(self.hidden_dim * 2, output_dim)
            self.dropout_layer = nn.Dropout(p=dropout)

        def forward(self, seq_batch: torch.Tensor) -> torch.Tensor:
            lstm_out, self.hidden = self.lstm(seq_batch, self.hidden)
            output = torch.cat(
                (lstm_out[-1, :, : self.hidden_dim], lstm_out[0, :, self.hidden_dim :]), 1
            )
            output = self.dropout_layer(output)
            output = self.linear1(output)
            output = self.dropout_layer(output)
            output = self.linear2(output)

            return output.view(self.batch_size, -1)

        def init_hidden(self) -> None:
            hidden = (
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim),
            )
            if self._is_cuda:
                hidden = tuple(ele.cuda() for ele in hidden)
            self.hidden = hidden

    def __init__(self, params: dict, torch_device: torch.device, logger: logging.Logger):
        """
        Format:

        ```
        model: linear model file

        model_params:
          input_dim: !!int
          hidden_dim: !!int
          batch_size: !!int
          output_dim: !!int
          num_layers: !!int
          dropout: !!float

        ar_states: ["list", "of", "state", "names"]
        ```
        """
        self._ar_states = params["ar_states"]
        self._is_cuda = torch_device.type == "cuda"
        self._logger = logger
        self._softmax_norm = nn.Softmax(dim=1)

        self._model = ModelManager.LSTMAR(
            params["model_params"]["input_dim"],
            params["model_params"]["hidden_dim"],
            params["model_params"]["batch_size"],
            params["model_params"]["output_dim"],
            params["model_params"]["num_layers"],
            self._is_cuda,
            params["model_params"]["dropout"],
        )
        self._model.load_state_dict(torch.load(params["model"], map_location=torch_device))
        self._model.cpu()
        if self._is_cuda:
            self._model.cuda()
        self._model.eval()

    def _init_hidden(self):
        self._model.init_hidden()

    def __call__(self, sample: torch.Tensor) -> Dict[str, float]:
        """
        Runs inference on a sample and returns a dict mapping state names to detection
        probabilities. Call to self._model() may raise an exception.
        """
        self._init_hidden()
        outs = self._model(sample)
        outs = self._softmax_norm(outs).view(-1).tolist()
        return {state: prob for state, prob in zip(self._ar_states, outs)}

    def empty_activities(self) -> Dict[str, float]:
        """Returns the same dict mapping as __call__, but with all zeros for results"""
        return {state: 0.0 for state in self._ar_states}


class SampleGenerator:
    """
    A SampleGenerator object will generate feature samples from a given media id and sample duration
    pair. It is initialized once, with shared information, and then any call to the returned
    instance will create a generator that yields samples.
    """

    def __init__(
        self,
        project_id: int,
        state_type: int,
        version: int,
        api: tator.api,
        client: boto3.client,
        work_dir: str,
        torch_device: torch.device,
        video_order: List[int],
        logger: logging.Logger,
    ):
        # The project ID where the media reside
        self._project = project_id

        # The state type to create from the inference
        self._state_type = state_type

        # The version to which the states will be uploaded
        self._version = version

        # The tator.api client
        self._api = api

        # The boto3 client
        self._client = client

        # The location to temporarily store feature files
        self._work_dir = work_dir

        # The device on which to put torch.Tensors
        self._torch_device = torch_device

        # The order in which to concatenate the videos, using their index in the
        # `multiview.media_files.ids` list. For example, a list of ids [2, 4, 8, 16] with a video
        # order [3, 2, 0, 1] would produce the id ordering [16, 8, 2, 4].
        self._video_order = video_order

        self._logger = logger

    @staticmethod
    def _process_stitch_map(stitch_map_str: Optional[str]) -> List[Tuple[int, int]]:
        """
        Takes the string contained in the `_stitch_map` attribute and turns it in to a list of
        tuples, where the first value of the tuple is the beginning of a frame gap and the second is
        the end of a frame gap. A frame gap is time between two videos that were stitched together
        where at least one frame of time was missing, relative to wall clock time.
        """
        if stitch_map_str is None:
            return []
        video_starts = []
        blank_starts = []

        try:
            stitch_map = json.loads(stitch_map_str.replace("`", '"'))
        except:
            return []

        for key, frame in stitch_map.items():
            if key.startswith("blank"):
                blank_starts.append(round(frame))
            else:
                video_starts.append(round(frame))

        video_starts.pop(0)

        assert len(blank_starts) == len(video_starts)
        return [(start, end) for start, end in zip(blank_starts, video_starts)]

    def _get_associated_media_ids(self, multiview_id: int) -> Tuple[List[int], int]:
        """
        Looks up a multiview from its ID and returns a tuple with the list of media ids including
        the multiview and all of its media file ids and the number of individual videos composing
        this multiview.
        """
        multiview = self._api.get_media(multiview_id)

        if multiview.media_files.ids is None:
            # These aren't the droids you're looking for
            return [], 0

        media_ids = list(multiview.media_files.ids)
        num_videos = len(media_ids)
        media_ids.append(multiview_id)
        return media_ids, num_videos

    @staticmethod
    def _get_max_frame(feature_filenames):
        return max(pd.read_hdf(fn, start=-1).iloc[-1].name for fn in feature_filenames)

    @staticmethod
    def _remove_feature_files(filenames):
        for filename in filenames:
            os.remove(filename)

    def _download_feature_files(self, media_ids, media_dict):
        """
        Downloads the previously extracted feature files for each video from s3. Returns the list of
        filenames in the desired order.
        """
        filenames = []
        try:
            for media_id in media_ids:
                self._logger.info(f"Loading features from {media_id}...")
                vid = media_dict[media_id]
                feature_str = str(vid.attributes["feature_s3"])
                try:
                    feature_s3 = json.loads(feature_str)
                except:
                    self._logger.info(
                        f"Could not read bucket/key information for {media_id}: '{feature_str}'",
                        exc_info=True,
                    )
                    raise

                key = feature_s3["key"]
                bucket = feature_s3["bucket"]
                feature_filename = os.path.join(self._work_dir, key)

                try:
                    self._logger.info(f"Downloading {feature_filename}")
                    self._client.download_file(bucket, key, feature_filename)
                    self._logger.info(f"{feature_filename} downloaded!")
                    filenames.append(feature_filename)
                except:
                    self._logger.warning(
                        f"Could not find '{key}' in bucket '{bucket}'", exc_info=True
                    )
                    raise
        except:
            # Clean up before raising
            self._remove_feature_files(filenames)

            raise

        return filenames

    def __call__(
        self, multiview_id: int, sample_size: int
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        """
        Makes a generator that yields a tuple containing a sample and the frame number of the
        beginning of the sample.
        """

        # Get media associated with given multiview id
        multiview = self._api.get_media(multiview_id)
        media_ids = list(multiview.media_files.ids)

        # Put them in the required order
        media_ids = [media_ids[idx] for idx in self._video_order]

        # Get media metadata
        media_dict = {
            vid.id: vid for vid in self._api.get_media_list(self._project, media_id=media_ids)
        }
        media_fps = [round(vid.fps) for vid in media_dict.values()]
        extant_state_frames = set(
            state.frame
            for state in self._api.get_state_list(
                self._project,
                type=self._state_type,
                media_id=[multiview_id],
                version=[self._version],
            )
        )

        # Parse frame gaps
        frame_gaps = [
            self._process_stitch_map(media_dict[media_id].attributes.get("_stitch_map"))
            for media_id in media_ids
        ]

        # Determine union of frame gaps across all media
        n_gaps = len(frame_gaps[0])
        if any(len(gaps) != n_gaps for gaps in frame_gaps):
            raise ValueError(f"Not all media contain the same number of frame gaps, aborting")

        global_frame_gaps = (
            [
                (min(gap[idx][0] for gap in frame_gaps), max(gap[idx][1] for gap in frame_gaps))
                for idx in range(n_gaps)
            ]
            if n_gaps > 0
            else []
        )

        # Download feature files from S3
        feature_filenames = self._download_feature_files(media_ids, media_dict)

        # Get global FPS
        if all(fps == media_fps[0] for fps in media_fps):
            fps = media_fps[0]
        else:
            fps = round(median(media_fps))
            self._logger.warning(f"Media FPS differ ({media_fps}), using median value {fps}.")

        # Generate samples
        progress_thresh = 10
        sample_start_frame = 0
        sample_size_frames = sample_size * fps
        sample_end_frame = sample_start_frame + sample_size_frames - 1
        max_frame = self._get_max_frame(feature_filenames)
        while sample_end_frame < max_frame:
            if global_frame_gaps:
                gap_start, gap_end = global_frame_gaps[0]

                if (
                    gap_start < sample_start_frame < gap_end
                    or gap_start < sample_end_frame < gap_end
                    or (sample_start_frame < gap_start and sample_end_frame > gap_end)
                ):
                    # If any part of the sample contains gap frames, skip to the end of the gap
                    sample_start_frame = gap_end
                    sample_end_frame = sample_start_frame + sample_size_frames - 1
                    global_frame_gaps.pop(0)
                    continue
                elif sample_start_frame > gap_end:
                    # If the sample start frame is beyond the end of the next frame gap (shouldn't
                    # happen), remove the frame gap and try again
                    global_frame_gaps.pop(0)
                    continue

            if sample_start_frame in extant_state_frames:
                # Idempotency check: an inferred state for this frame range already exists, skip
                # it.
                logger.warning(
                    f"Found extant state for frame {sample_start_frame} in {multiview_id}, skipping inference..."
                )
            else:
                sample_tensor = self._tensor_from_features(
                    feature_filenames, sample_start_frame, sample_end_frame
                )

                yield sample_start_frame, sample_tensor

            # Log progress periodically
            progress = sample_start_frame / max_frame * 100
            if progress > progress_thresh:
                progress_thresh += 10
                self._logger.info(f"Progress {progress:.2f}%")

            # Update start and end frames for next iteration
            sample_start_frame = sample_end_frame + 1
            sample_end_frame = sample_start_frame + sample_size_frames - 1

        # Clean up after sample generation
        self._remove_feature_files(feature_filenames)

    def _tensor_from_features(self, feature_filenames, sample_start_frame, sample_end_frame):
        # Concatenate the samples from each video into a single feature vector per frame
        condition = [f"index in {list(range(sample_start_frame, sample_end_frame + 1))}"]
        try:
            sample_parts = [
                pd.read_hdf(feature_filename, where=condition)
                for feature_filename in feature_filenames
            ]
        except:
            self._logger.info(f"Problem subsampling features.", exc_info=True)
            raise

        sample = pd.concat(sample_parts, axis=1)
        sample.columns = list(range(len(sample.columns)))
        sample_batch = [torch.Tensor(sample.iloc[idx].to_numpy()) for idx in range(len(sample))]
        return (
            torch.cat(sample_batch)
            .view(len(sample_batch), 1, -1)
            .to(self._torch_device, non_blocking=True)
        )


def main(
    access_key: str,
    secret_key: str,
    s3_bucket: str,
    endpoint_url: str,
    attribute_name: str,
    upload_version: int,
    work_dir: str,
    project_id: int,
    state_type: int,
    host: str,
    token: str,
    model_config_params: dict,
    multiview_ids: List[int],
    sample_size: int,
    video_order: List[int],
    logger: logging.Logger,
    gpu_num: int = 0,
):

    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_num))
        torch_device_str = f"cuda:{gpu_num}"
    else:
        torch_device_str = "cpu"
    torch_device = torch.device(torch_device_str)

    # Initialize tator and S3 clients
    api = tator.get_api(host, token)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    with torch.no_grad():

        # Initialize models and sample generator
        model_manager = ModelManager(model_config_params, torch_device, logger)
        sample_generator = SampleGenerator(
            project_id,
            state_type,
            upload_version,
            api,
            client,
            work_dir,
            torch_device,
            video_order,
            logger,
        )

        desired_num_videos = len(video_order)

        for multiview_id in multiview_ids:
            logger.info(f"Starting inference on {multiview_id}")
            media_ids, actual_num_videos = sample_generator._get_associated_media_ids(multiview_id)

            if actual_num_videos != desired_num_videos:
                logger.warning(
                    f"Multiview {multiview_id} contains {actual_num_videos}, but the desired number is {desired_num_videos}, skipping inference..."
                )
                continue

            state_spec_list = []
            state_header = {
                "project": project_id,
                "type": state_type,
                "media_ids": media_ids,
                "version": upload_version,
            }

            # Perform activity inference on each sample of this multiview
            samples = sample_generator(multiview_id, sample_size)
            frame = -1
            sample = None
            while True:
                try:
                    # Use the `samples` generator with `next()` in a while True in order to put it
                    # inside of a `try...except` block, since it is possible for the generator to
                    # raise an exception.
                    frame, sample = next(samples)
                except StopIteration:
                    # Expected end of generator
                    break
                except:
                    logger.error(
                        f"Failed getting sample from {multiview_id}; last successful sample from frame {frame}",
                        exc_info=True,
                    )
                    break

                try:
                    activities = model_manager(sample)
                except:
                    logger.error(
                        f"Caught exception during inference, skipping sample in {multiview_id} starting at frame {frame}",
                        exc_info=True,
                    )
                    continue

                state = {**state_header, **activities}
                state["frame"] = frame

                if any(isnan(value) for value in activities.values()):
                    logger.error(
                        f"Activities in {multiview_id} contain a NaN, discarding result: {state}"
                    )
                    state.update(model_manager.empty_activities())

                state_spec_list.append(state)

            # Upload all inferred states
            n_states = len(state_spec_list)
            logger.info(f"Generated {n_states} for {multiview_id}, uploading...")
            states_uploaded = 0
            for idx in range(MAX_CHUNKED_CREATE_RETRIES):
                try:
                    for response in tator.util.chunked_create(
                        api.create_state_list, project_id, state_spec=state_spec_list
                    ):
                        logger.info(response.message)
                        states_uploaded += len(response.id)
                except:
                    logger.error(
                        f"Failed attempt {idx + 1} of {MAX_CHUNKED_CREATE_RETRIES} state chunked create for {multiview_id}, "
                        f"deleting {states_uploaded} states{' and trying again' if idx + 1 < MAX_CHUNKED_CREATE_RETRIES else ''}...",
                        exc_info=True,
                    )
                    try:
                        api.delete_state_list(
                            project_id, media_id=[multiview_id], version=[upload_version]
                        )
                    except:
                        logger.error(
                            f"Failed to delete uploaded states on {multiview_id} (version {upload_version}), needs manual intervention!",
                            exc_info=True,
                        )
                        break
                else:
                    logger.info(
                        f"{states_uploaded} (of {n_states}) states for {multiview_id} uploaded successfully!"
                    )
                    break
            else:
                logger.error(
                    f"Failed to upload states for multiview {multiview_id} after {MAX_CHUNKED_CREATE_RETRIES}, moving on..."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for evaluating Activity Recognition pipeline"
    )
    tator.get_parser(parser)
    parser.add_argument("--access-key", type=str, required=True)
    parser.add_argument("--secret-key", type=str, required=True)
    parser.add_argument("--s3-bucket", type=str, required=True)
    parser.add_argument("--endpoint-url", type=str, required=True)
    parser.add_argument("--attribute-name", type=str, required=True)
    parser.add_argument("--upload-version", type=int, required=True)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--project-id", type=int, required=True)
    parser.add_argument("--state-type", type=int, required=True)
    parser.add_argument(
        "--model-config-file", help="File containing runtime parameters", type=str, required=True
    )
    parser.add_argument(
        "--multiview-ids", help="List of multiviews to process", nargs="*", type=int, required=True
    )
    parser.add_argument(
        "--sample-size", help="The length of a sample in seconds", type=int, required=True
    )
    parser.add_argument(
        "--video-order",
        help="The order in which the video features should be composed",
        nargs="*",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    with open(args.model_config_file, "r") as fp:
        model_config_params = yaml.load(fp)

    log_filename = os.path.join(args.work_dir, "ar_inference.log")
    logging.basicConfig(
        handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.INFO,
    )
    logger = logging.getLogger(__file__)

    main(
        access_key=args.access_key,
        secret_key=args.secret_key,
        s3_bucket=args.s3_bucket,
        endpoint_url=args.endpoint_url,
        attribute_name=args.attribute_name,
        upload_version=args.upload_version,
        work_dir=args.work_dir,
        project_id=args.project_id,
        state_type=args.state_type,
        host=args.host,
        token=args.token,
        model_config_params=model_config_params,
        multiview_ids=args.multiview_ids,
        sample_size=args.sample_size,
        video_order=args.video_order,
        logger=logger,
    )
