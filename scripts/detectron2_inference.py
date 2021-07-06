import detectron2
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

import torchvision

from PIL import Image

from functools import partial
import multiprocessing as mp
import signal
import os
import glob
import pickle
import json
import cv2
import random
import torch
import numpy as np
import pandas as pd
import time
from typing import List, Optional, Union
from utils.frame_reader import FrameReaderMgrBase


if torch.cuda.is_available():
    torch.set_default_tensor_type("torch.cuda.HalfTensor")


FRAME_TIMEOUT = 5  # seconds
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
        augmentation: T.Augmentation,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._augmentation = augmentation

    def _format_img(self, img, frame_num):
        img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
        img = self._augmentation.get_transform(img).apply_image(img)
        _, h, w = img.shape
        return {"image": img, "height": h, "width": w, "frame_num": frame_num}


def parse_args():
    parser = argparse.ArgumentParser(description="Testing script for testing video data.")
    parser.add_argument("model_config", help="Path to inference config file.")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--gpu", help="Id of the GPU to use (as reported by nvidia-smi).")
    parser.add_argument(
        "--score-threshold", help="Threshold to filter detections", default=0.7, type=float
    )
    parser.add_argument(
        "--batch-size", help="batch size for frames to process at a time", default=8, type=int
    )
    parser.add_argument(
        "--nms-threshold", help="threshold for NMS routine to suppress", default=0.55, type=float
    )

    return parser.parse_args()


if __name__ == "__main__":
    # parse arguments
    args = parse_args()
    """
    This is where the model is instantiated. There is a LOT of nested arguments in these yaml files, and the merging of baseline defaults plus
    dataset specific parameters. I recommend spending a decent chunk of time trying to delve into some of the parameters.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.merge_from_file(
        "/mnt/md0/Projects/Fathomnet/Training_Files/2021-06-29-Detectron/detectron_files/fathomnet_config.yaml"
    )
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.WEIGHTS = ""  # path to the model file

    model = build_model(cfg)  # returns a torch.nn.Module
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    """
    separate NMS layer
    """
    model_nms = torchvision.ops.nms

    aug = T.ResizeShortestEdge(
        [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
    )

    frame_reader = FrameReaderMgr(augmentation=aug)
    results = []

    with frame_reader(args.video_path) as frame_queue, done_event:
        while True:
            st = time.time()
            batch = []
            for idx in range(args.batch_size):
                try:
                    image = frame_queue.get(timeout=FRAME_TIMEOUT)
                    batch.append(image)
                except:
                    print("timed out")
                    done_event.set()
                    break
                print("Elapsed time pre-process = {}".format(time.time() - st))

            model_outputs = model(batch)

            # TODO Find the right spot for the nms call. Either here or at the end with post-proessing. Have to make sure you do it
            # by element, because model_outputs is a list of instances.
            for elem in model_outputs:
                elem["instances"] = elem["instances"][
                    model_nms(
                        elem["instances"].pred_boxes.tensor,
                        elem["instances"].scores,
                        args.nms_threshold,
                    )
                    .to("cpu")
                    .tolist()
                ]
                results.append(elem)

            print(f"Elapsed time model = {time.time() - st}")

    if results:
        # WARNING
        # write output - This is super fragile code you'll want to change
        out_name = re.split(".mov", args.video_path.split("/")[-1], flags=re.IGNORECASE)[0]
        print(out_name)
        try:
            # json.dump(results, open('{}_bbox_results.json'.format(out_name), 'w'), indent=4)
            pickle.dump(
                results,
                open("{}_bbox_results_{}.pickle".format(out_name, args.score_threshold), "wb"),
            )
        except:
            pickle.dump(results, open("default_bbox_results.pickle", "wb"))
            # json.dump(results, open('default_bbox_results.json', 'w'), indent=4)

    logger.info("Finished")
    sys.exit()
