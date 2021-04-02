from IPython import embed

import argparse
import json
import logging
import os
import pandas as pd
import pickle
from statistics import median
from typing import Generator, List, Tuple
import yaml

import boto3
import torch
import torch.nn as nn
import torchvision
import numpy as np

import tator

if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


AR_STATES = [
    "Hold Loading Score",
    "Net In / Out Score",
    "Offloading Score",
    "Sorting Score",
    "Background",
    "Net Out / Sort Score",
]


logging.basicConfig(
    handlers=[logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


class LSTMAR(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        batch_size: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0,
    ):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
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

    def init_hidden(self):
        return (
            (
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda(),
            )
            if torch.cuda.is_available()
            else (
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim),
            )
        )

    def forward(self, seq_batch: torch.Tensor):
        lstm_out, self.hidden = self.lstm(seq_batch, self.hidden)
        output = torch.cat(
            (lstm_out[-1, :, : self.hidden_dim], lstm_out[0, :, self.hidden_dim :]), 1
        )
        output = self.dropout_layer(output)
        output = self.linear1(output)
        output = self.dropout_layer(output)
        output = self.linear2(output)

        return output.view(self.batch_size, -1)


class LSTMARNOLIN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        batch_size: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0,
    ):

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout)
        self.linear = nn.Linear(self.hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self):
        return (
            (
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim).cuda(),
            )
            if torch.cuda.is_available()
            else (
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            )
        )

    def forward(self, seq_batch: torch.Tensor):
        lstm_out, self.hidden = self.lstm(seq_batch, self.hidden)
        output = self.linear(lstm_out[-1].view(self.batch_size, -1))
        output = self.sigmoid(output)

        return output.view(self.batch_size, -1)


def init_models(params: dict, torch_device: torch.device) -> Tuple[LSTMARNOLIN, LSTMAR]:
    """
    Format:

    ```
    nolin_model: nonlinear model file
    lin_model: linear model file

    model_params:
      input_dim: !!int
      hidden_dim: !!int
      batch_size: !!int
      output_dim: !!int
      num_layers: !!int
      dropout: !!float
    ```
    """

    nolin_model = LSTMARNOLIN(
        params["model_params"]["input_dim"],
        params["model_params"]["hidden_dim"],
        params["model_params"]["batch_size"],
        params["model_params"]["output_dim"],
        params["model_params"]["num_layers"],
        params["model_params"]["dropout"],
    )
    nolin_model.load_state_dict(torch.load(params["nolin_model"], map_location=torch_device))
    nolin_model.cpu()
    if torch.cuda.is_available():
        nolin_model.cuda()
    nolin_model.eval()

    lin_model = LSTMAR(
        params["model_params"]["input_dim"],
        params["model_params"]["hidden_dim"],
        params["model_params"]["batch_size"],
        params["model_params"]["output_dim"],
        params["model_params"]["num_layers"],
        params["model_params"]["dropout"],
    )
    lin_model.load_state_dict(torch.load(params["lin_model"], map_location=torch_device))
    lin_model.cpu()
    if torch.cuda.is_available():
        lin_model.cuda()
    lin_model.eval()

    return nolin_model, lin_model


class SampleGenerator:
    def __init__(
        self,
        project_id: int,
        api: tator.api,
        client: boto3.client,
        work_dir: str,
        torch_device: torch.device,
        video_order: List[int],
    ):
        # The project ID where the media reside
        self._project = project_id

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

    @staticmethod
    def _process_stitch_map(stitch_map_str: str) -> List[Tuple[int, int]]:
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

    def get_associated_media_ids(self, multiview_id: int) -> List[int]:
        multiview = self._api.get_media(multiview_id)
        media_ids = list(multiview.media_files.ids)
        media_ids.append(multiview_id)
        return media_ids

    def __call__(
        self, multiview_id: int, sample_size: int
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        # Get media associated with given multiview id
        multiview = self._api.get_media(multiview_id)
        media_ids = list(multiview.media_files.ids)
        media_ids = [media_ids[idx] for idx in self._video_order]
        media_dict = {
            vid.id: vid for vid in self._api.get_media_list(self._project, media_id=media_ids)
        }
        media_fps = [vid.fps for vid in media_dict.values()]

        # Download feature files from S3 and parse frame gaps
        feature_files = []
        frame_gaps = [
            self._process_stitch_map(media_dict[media_id].attributes["_stitch_map"])
            for media_id in media_ids
        ]
        for media_id in media_ids:
            vid = media_dict[media_id]
            try:
                feature_s3 = json.loads(vid.attributes["feature_s3"])
            except:
                logger.info(f"Could not load features for {vid.id}, aborting", exc_info=True)
                embed()
                raise
            feature_files.append(os.path.join(self._work_dir, feature_s3["key"]))
            logger.info(f"Downloading {feature_files[-1]}")
            if not os.path.isfile(feature_files[-1]):
                self._client.download_file(
                    feature_s3["bucket"], feature_s3["key"], feature_files[-1]
                )
            logger.info(f"{feature_files[-1]} downloaded!")

        # Determine union of frame gaps across all media
        n_gaps = len(frame_gaps[0])
        assert all(len(gaps) == n_gaps for gaps in frame_gaps)

        global_frame_gaps = [
            (max(gap[idx][0] for gap in frame_gaps), max(gap[idx][1] for gap in frame_gaps))
            for idx in range(n_gaps)
        ]

        # Load feature files
        features = [pd.read_hdf(ff) for ff in feature_files]
        max_frame = min(feat.iloc[-1].name for feat in features)

        # Generate samples
        if all(fps == media_fps[0] for fps in media_fps):
            fps = media_fps[0]
        else:
            logger.warning(f"Media FPS differs ({media_fps}), using median value {median_fps}")
            fps = round(median(media_fps))
        sample_size_frames = sample_size * fps
        sample_start_frame = 0
        while True:
            sample_end_frame = sample_start_frame + sample_size_frames - 1

            if global_frame_gaps:
                next_frame_gap = global_frame_gaps[0]

                if (
                    next_frame_gap[0] < sample_start_frame < next_frame_gap[1]
                    or next_frame_gap[0] < sample_end_frame
                ):
                    # If the start frame is too close to or contained in the next frame gap, skip to
                    # the end of it
                    sample_start_frame = next_frame_gap[1]
                    global_frame_gaps.pop(0)
                    continue
                elif sample_start_frame > next_frame_gap[1]:
                    # If the sample start frame is beyond the end of the next frame gap (shouldn't
                    # happen), remove the frame gap and try again
                    global_frame_gaps.pop(0)
                    continue
            else:
                # If the end of the sample is past the end of the shortest video, sample generation
                # is complete
                if max_frame < sample_end_frame:
                    break

            # Concatenate the samples from each video into a single feature vector per frame
            sample_parts = []
            for idx, feature in enumerate(features):
                indices = list(feature.index)
                start_frame = sample_start_frame
                end_frame = sample_end_frame

                try:
                    subsample = feature.loc[start_frame:end_frame]
                except:
                    logger.info(f"Problem subsampling feature no. {idx}", exc_info=True)
                    raise
                else:
                    sample_parts.append(subsample)

            sample = pd.concat(sample_parts, axis=1)
            sample.columns = list(range(len(sample.columns)))
            sample_batch = [torch.Tensor(sample.iloc[idx].to_numpy()) for idx in range(len(sample))]
            sample_tensor = (
                torch.cat(sample_batch)
                .view(len(sample_batch), 1, -1)
                .to(self._torch_device, non_blocking=True)
            )
            yield sample_start_frame, sample_tensor
            sample_start_frame = sample_end_frame + 1


def main(
    access_key: str,
    secret_key: str,
    s3_bucket: str,
    endpoint_url: str,
    attribute_name: str,
    work_dir: str,
    project_id: int,
    state_type: int,
    host: str,
    token: str,
    model_config_params: dict,
    multiview_ids: List[int],
    sample_size: int,
    video_order: List[int],
    gpu_num: int = 0,
):

    if torch.cuda.is_available():
        torch.cuda.set_device(int(gpu_num))
        torch_device = torch.device(f"cuda:{gpu_num}")
    else:
        torch_device = torch.device("cpu")

    api = tator.get_api(host, token)
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    with torch.no_grad():
        softmax_norm = torch.nn.Softmax()
        nolin_model, lin_model = init_models(model_config_params, torch_device)
        sample_generator = SampleGenerator(
            project_id, api, client, work_dir, torch_device, video_order
        )

        for multiview_id in multiview_ids:
            logger.info(f"Starting inference on {multiview_id}")
            media_ids = sample_generator.get_associated_media_ids(multiview_id)

            state_spec_list = []
            for frame, sample in sample_generator(multiview_id, sample_size):
                logger.info(f"  Frame {frame}")
                lin_model.hidden = lin_model.init_hidden()
                nolin_model.hidden = nolin_model.init_hidden()

                try:
                    lin_out = lin_model(sample)
                    nolin_out = nolin_model(sample)
                except:
                    logger.info(
                        f"Model failed to process sample starting at frame {frame}", exc_info=True
                    )
                    raise

                lin_out = softmax_norm(lin_out)
                lin_out = lin_out.view(-1).tolist()
                nolin_out = nolin_out.view(-1).tolist()

                activities = [
                    nolin_out[0],
                    np.maximum(lin_out[1], lin_out[5]),
                    nolin_out[2],
                    np.maximum(lin_out[3], lin_out[5]),
                    nolin_out[4],
                ]

                state = {
                    "project": project_id,
                    "type": state_type,
                    "frame": frame,
                    "media_ids": media_ids,
                }

                state.update(zip(AR_STATES, activities))
                state_spec_list.append(state)

            n_states = len(state_spec_list)
            responses = []
            for response in tator.util.chunked_create(
                api.create_state_list, project_id, state_spec=state_spec_list
            ):
                responses += response.id
                print(f"Progress {len(responses) / n_states * 100}%")


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

    main(
        access_key=args.access_key,
        secret_key=args.secret_key,
        s3_bucket=args.s3_bucket,
        endpoint_url=args.endpoint_url,
        attribute_name=args.attribute_name,
        work_dir=args.work_dir,
        project_id=args.project_id,
        state_type=args.state_type,
        host=args.host,
        token=args.token,
        model_config_params=model_config_params,
        multiview_ids=args.multiview_ids,
        sample_size=args.sample_size,
        video_order=args.video_order,
    )
