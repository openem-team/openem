import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from typing import List

from detectron2.structures import BoxMode
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.modeling import build_model
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
import pandas as pd
import torch
import torchvision

from utils.frame_reader import FrameReaderMgrBase
from utils.file_downloader import FileDownloader
import tator


log_filename = "detectron2_inference.log"


logging.basicConfig(
    handlers=[logging.FileHandler(log_filename, mode="w"), logging.StreamHandler()],
    format="%(asctime)s %(levelname)s:%(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


class FrameReaderMgr(FrameReaderMgrBase):
    def __init__(
        self,
        *,
        augmentation: T.Augmentation,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._augmentation = augmentation

    def _format_img(self, img, frame_num):
        h, w = img.shape[:2]
        img = self._augmentation.get_transform(img).apply_image(img)
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        return {"image": img, "height": h, "width": w, "frame_num": frame_num}


class LocalizationGenerator:
    def __init__(self, model_nms, nms_threshold, localization_type):
        self._model_nms = model_nms
        self._nms_threshold = nms_threshold
        self._localization_type = localization_type

    def __call__(self, element, frame, media_id):
        """
        Yields `LocalizationSpec`s from the model detections in a video frame.
        """
        element["instances"] = element["instances"][
            self._model_nms(
                element["instances"].pred_boxes.tensor,
                element["instances"].scores,
                self._nms_threshold,
            )
            .to("cpu")
            .tolist()
        ]

        instance_dict = element["instances"].get_fields()
        pred_boxes = instance_dict["pred_boxes"]
        scores = instance_dict["scores"]
        pred_classes = instance_dict["pred_classes"]

        # TODO check attribute names and determine if they should be dynamic
        # yield LocalizationSpec
        for box, score, cls in zip(pred_boxes, scores, pred_classes):
            x1, y1, x2, y2 = box.tolist()
            yield {
                "type": self._localization_type,
                "media_id": media_id,
                "frame": frame,
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "Species": cls,
                "Score": score,
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Testing script for testing video data.")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--inference-config",
        help="Path to inference config file.",
        # TODO remove default here
        default="/mnt/md0/Projects/Fathomnet/Training_Files/2021-06-29-Detectron/detectron_files/fathomnet_config.yaml",
    )
    parser.add_argument(
        "--builtin-model-config",
        help="Path to built-in model config file.",
        # TODO remove default here
        default="COCO-Detection/retinanet_R_50_FPN_3x.yaml",
    )
    parser.add_argument(
        "--model-weights",
        help="Path to the trained model weights",
        # TODO remove default here
        default="/home/hugh/mycode/detectron/out/model_0076543.pth",
    )
    parser.add_argument(
        "--gpu", help="Id of the GPU to use (as reported by nvidia-smi).", default=0, type=int
    )
    parser.add_argument(
        "--score-threshold", help="Threshold to filter detections", default=0.7, type=float
    )
    parser.add_argument(
        "--batch-size", help="batch size for frames to process at a time", default=4, type=int
    )
    parser.add_argument(
        "--nms-threshold", help="threshold for NMS routine to suppress", default=0.55, type=float
    )
    parser.add_argument("--media-ids", help="The ids of the media to process", nargs="+", type=int)
    parser.add_argument(
        "--localization-type", help="The id of the localization type to generate", type=int
    )
    parser.add_argument("--host", type=str, help="Tator host to use")
    parser.add_argument("--token", type=str, help="Token to use for tator.")
    parser.add_argument(
        "--work-dir", type=str, help="The name of the directory to use for local storage"
    )

    return parser.parse_args()


def main(
    *,
    inference_config: str,
    builtin_model_config: str,
    model_weights: str,
    video_path: str,
    batch_size: int,
    nms_threshold: float,
    score_threshold: float,
    gpu: int,
    media_ids: List[int],
    localization_type: int,
    host: str,
    token: str,
    work_dir: str,
):
    # Download associated media
    api = tator.get_api(host=host, token=token)
    download = FileDownloader(work_dir, api)
    media_paths = download(media_ids)

    # Instantiate the model
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(builtin_model_config))
    cfg.merge_from_file(inference_config)
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3  # TODO magic number
    cfg.MODEL.WEIGHTS = model_weights
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg)  # returns a torch.nn.Module
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()

    # Separate NMS layer
    model_nms = torchvision.ops.nms
    aug = T.ResizeShortestEdge(
        short_edge_length=[cfg.INPUT.MIN_SIZE_TEST],
        max_size=cfg.INPUT.MAX_SIZE_TEST,
        sample_style="choice",
    )

    localization_generator = LocalizationGenerator(model_nms, nms_threshold, localization_type)
    frame_reader = FrameReaderMgr(augmentation=aug)
    results = []

    for media_id, media_path in zip(media_ids, media_paths):
        with frame_reader(media_path):
            logger.info(f"Generating detections for {media_id}")
            st = time.time()
            while True:
                try:
                    batch = frame_reader.get_frames(batch_size)
                except:
                    break
                else:
                    frames = [ele["frame_num"] for ele in batch]

                with torch.no_grad():
                    model_outputs = model(batch)

                results.extend(
                    loc
                    for frame_detections, frame in zip(model_outputs, frames)
                    for loc in localization_generator(frame_detections, frame, media_id)
                )


        if results:
            created_ids = []
            for response in tator.util.chunked_create(
                tator_api.create_localization_list, project, localization_spec=results
            ):
                created_ids += response.id

            n_requested = len(results)
            n_created = len(created_ids)
            if n_created == n_requested:
                logger.info(f"Created {n_created} localizations for {media_id}!")
            else:
                logger.warning(
                    f"Requested the creation of {n_requested} localizations, but only {n_created} were created for {media_id}"
                )
        else:
            logger.info(f"No detections for media {media_id}")


if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    main(**vars(args))
    logger.info("Finished")
