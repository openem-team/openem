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

def format_img(img,aug,frame_num):
    img = torch.as_tensor(img.astype("float32").transpose(2, 0, 1))
    img = aug.get_transform(img).apply_image(img)
    _,h,w = img.shape
    return {"image" : img, "height" : h, "width" : w, "frame_num" : frame_num}

def read_frames(img_path, raw_frame_queue, stop_event):
    ok = True
    vid = cv2.VideoCapture(img_path)
    if cv2.__version__ >= "3.2.0":
        vid_len = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
    else:
        vid_len = int(vid.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.cv.CV_CAP_PROP_FPS)
    frame_num = 0
    while ok and not stop_event.is_set():
        if not raw_frame_queue.full():
            ok,img = vid.read()
            if ok:
                raw_frame_queue.put((img, frame_num))
                frame_num += 1
    stop_event.set()
    print("This thread should exit now read frames")
    return

def enqueue_frames(raw_frame_queue, processed_frame_queue, stop_event, aug):
    while not stop_event.is_set():
        try:
            img,frame_num = raw_frame_queue.get(timeout=5)
            X = format_img(img,aug,frame_num)
            processed_frame_queue.put((X,frame_num))
        except:
            stop_event.set()
            print("This thread should exit now")
    return

def signal_handler(stop_event, frame_stop_event,signal_received, frame):
    # Handle any cleanup here
    stop_event.set()
    frame_stop_event.set()
    print('SIGINT or CTRL-C detected. Exiting gracefully')
    for i in range(10):
        print(f'Shutting down in {10-i}')
        time.sleep(1)

def parse_args():
    parser = argparse.ArgumentParser(
            description='Testing script for testing video data.')
    parser.add_argument('model_config', help='Path to inference config file.')
    parser.add_argument('video_path', 
            help='Path to video file')
    parser.add_argument('--gpu', 
            help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold', 
            help='Threshold to filter detections', 
            default=0.7, 
            type=float)
    parser.add_argument('--batch-size', 
            help='batch size for frames to process at a time', 
            default=8, 
            type=int)
    parser.add_argument('--nms-threshold', 
        help='threshold for NMS routine to suppress', 
        default=0.55, 
        type=float)

    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    '''
    This is where the model is instantiated. There is a LOT of nested arguments in these yaml files, and the merging of baseline defaults plus
    dataset specific parameters. I recommend spending a decent chunk of time trying to delve into some of the parameters.
    '''
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
    cfg.merge_from_file("/mnt/md0/Projects/Fathomnet/Training_Files/2021-06-29-Detectron/detectron_files/fathomnet_config.yaml")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.WEIGHTS = '' # path to the model file

    model = build_model(cfg)  # returns a torch.nn.Module
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)

    '''
    separate NMS layer
    '''
    model_nms = torchvision.ops.nms

    aug = T.ResizeShortestEdge(
                [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)

    raw_queue = mp.Queue(50)
    frame_queue = mp.Queue(50)
    stop_event = mp.Event()
    frame_stop_event = mp.Event()
    signal.signal(
        signal.SIGINT, 
        partial(signal_handler, stop_event, frame_stop_event))
    p = mp.Process(target=read_frames, args=(args.video_path,raw_queue,stop_event))
    p.daemon = True
    p.start()
    for num in range(8):
        p = mp.Process(target=enqueue_frames, args=(raw_queue, frame_queue, frame_stop_event, aug))
        p.daemon = True
        p.start()
    id_ = 1
    results = []

    while True:
        st = time.time()
        batch = []
        for idx in range(args.batch_size):
            try:
                image = frame_queue.get(timeout=5)
                batch.append(image)
            except:
                print("timed out")
                break
            print('Elapsed time pre-process = {}'.format(time.time() - st))

        model_outputs = model(batch)

        #TODO Find the right spot for the nms call. Either here or at the end with post-proessing. Have to make sure you do it
        #by element, because model_outputs is a list of instances.
        for elem in model_outputs:
            elem["instances"] = elem["instances"][model_nms(elem["instances"].pred_boxes.tensor, elem["instances"].scores, args.nms_threshold).to("cpu").tolist()]
            results.append(elem)

        print('Elapsed time model = {}'.format(time.time() - st))

    if len(results):
        #WARNING
        # write output - This is super fragile code you'll want to change
        out_name = re.split(".mov",args.video_path.split('/')[-1],flags=re.IGNORECASE)[0]
        print(out_name)
        try:
            #json.dump(results, open('{}_bbox_results.json'.format(out_name), 'w'), indent=4)
            pickle.dump(results,open('{}_bbox_results_{}.pickle'.format(out_name, args.score_threshold),'wb'))
        except:
            pickle.dump(results,open('default_bbox_results.pickle','wb'))
            #json.dump(results, open('default_bbox_results.json', 'w'), indent=4)

    frame_stop_event.set()
    stop_event.set()
    print("Finished")
    sys.exit()