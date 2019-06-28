#!/usr/bin/env python3

import sys
sys.path.append("../train")
from collections import namedtuple
import math
import configparser
from openem_train.ssd.ssd_dataset import SSDDataset
from openem_train.util import config_interface
import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import pandas as pd

def checkFile(video_id : str, ssd, config):
    # Grab first sample
    for detection in ssd.detections[video_id]:
        print("Checking detection = {}".format(detection))
        cfg=ssd.get_config(detection, True)

        crop, targets = ssd.generate_xy(cfg)
        if len(targets) > 0:
            boxX=np.array([targets[0][0],targets[0][2]])*config.detect_width()
            boxY=np.array([targets[0][1],targets[0][3]])*config.detect_height()

            length=math.hypot(boxX[1]-boxX[0],boxY[1]-boxY[0])
            print(f"{boxX[0]},{boxY[0]} -- {boxX[1]},{boxY[1]}")
            print(f"length={length}")
            print("NOTICE: You should see a fish.")
            plt.imshow(crop/255.0)
            plt.plot(boxX,boxY,'or')
            plt.show()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="hostTrain.ini")
    args=parser.parse_args()

    video_id="00WK7DR6FyPZ5u3A"
    #video_id="0QAlqRiUad7xcB9k"
    config=config_interface.ConfigInterface(args.config)
    lengths=pd.read_csv(config.length_path())

    # Aren't using bbox util for this test
    ssd=SSDDataset(config, None)

    for _,annotation in lengths.iterrows():
        checkFile(annotation['video_id'], ssd, config)
