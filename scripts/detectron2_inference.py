import argparse
import json
import logging
import multiprocessing as mp
import os
import time

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


def parse_args():
    parser = argparse.ArgumentParser(description="Testing script for testing video data.")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument(
        "--inference_config",
        help="Path to inference config file.",
        # TODO remove default here
        default="/mnt/md0/Projects/Fathomnet/Training_Files/2021-06-29-Detectron/detectron_files/fathomnet_config.yaml",
    )
    parser.add_argument(
        "--builtin_model_config",
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

    return parser.parse_args()


def generate_detections(element, frame, model_nms, nms_threshold):
    element["instances"] = element["instances"][
        model_nms(
            element["instances"].pred_boxes.tensor,
            element["instances"].scores,
            nms_threshold,
        )
        .to("cpu")
        .tolist()
    ]

    instance_dict = element["instances"].get_fields()
    pred_boxes = instance_dict["pred_boxes"]
    scores = instance_dict["scores"]
    pred_classes = instance_dict["pred_classes"]

    # yield LocalizationSpec-ish
    # TODO add media_id, type, species annotation, score annotation
    for box, score, cls in zip(pred_boxes, scores, pred_classes):
        x1, y1, x2, y2 = box.tolist()
        yield {"frame": frame, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}


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
):
    """
    This is where the model is instantiated. There is a LOT of nested arguments in these yaml files,
    and the merging of baseline defaults plus dataset specific parameters. I recommend spending a
    decent chunk of time trying to delve into some of the parameters.
    """
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

    """
    separate NMS layer
    """
    model_nms = torchvision.ops.nms
    aug = T.ResizeShortestEdge(
        short_edge_length=[cfg.INPUT.MIN_SIZE_TEST],
        max_size=cfg.INPUT.MAX_SIZE_TEST,
        sample_style="choice",
    )

    frame_reader = FrameReaderMgr(augmentation=aug)
    results = []
    start_frame = 0

    with frame_reader(video_path):
        while True:
            st = time.time()
            frames = range(start_frame, start_frame + batch_size)
            try:
                batch = frame_reader.get_frames(batch_size)
            except:
                break
            logger.info(f"Elapsed time pre-process = {time.time() - st}")

            with torch.no_grad():
                model_outputs = model(batch)

            # TODO Find the right spot for the nms call. Either here or at the end with
            # post-proessing. Have to make sure you do it by element, because model_outputs is a
            # list of instances.
            results.extend(
                det
                for ele, frame in zip(model_outputs, frames)
                for det in generate_detections(ele, frame, model_nms, nms_threshold)
            )

            logger.info(f"Elapsed time model = {time.time() - st}")
            start_frame += batch_size

    if results:
        logger.info("got results")
        # WARNING
        # write output - This is super fragile code you'll want to change
        out_name = os.path.splitext(os.path.split(video_path)[1])[0]
        try:
            with open(f"{out_name}_bbox_results_{score_threshold}.json", "w") as fp:
                json.dump(results, fp, indent=4)
        except:
            with open("default_bbox_results.json", "w") as fp:
                json.dump(results, fp, indent=4)
    else:
        logger.info("no results")


if __name__ == "__main__":
    # parse arguments
    args = parse_args()

    main(**vars(args))
    logger.info("Finished")
