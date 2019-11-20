from openem.models import ImageModel
from openem.models import Preprocessor

import cv2
import numpy as np
import tensorflow as tf

from collections import namedtuple
import csv

Detection=namedtuple('Detection', ['location',
                                   'confidence',
                                   'species',
                                   'frame',
                                   'video_id'])

# Bring in SSD detector to top-level
from openem.Detect.SSD import SSDDetector

class IO:
    def from_csv(filepath_like):
        detections=[]
        with open(filepath_like, 'r') as csv_file:
            reader = csv.DictReader(csv_file)

            last_idx = -1
            for row in reader:
                location=np.array([float(row['x']),
                                   float(row['y']),
                                   float(row['w']),
                                   float(row['h'])])
                item = Detection(location=location,
                                 confidence=float(row['detection_conf']),
                                 species=int(float(row['detection_species'])),
                                 frame=int(row['frame']),
                                 video_id=row['video_id'])

                frame_num = int(float(row['frame']))
                if last_idx == frame_num:
                    detections[last_idx].append(item)
                else:
                    # Add empties
                    for _ in range(frame_num-1-last_idx):
                        detections.append([])
                    detections.append([item])
                    last_idx = frame_num
        return detections

