import numpy as np
import pandas as pd
import json
import os
import pickle
from typing import List, Dict
from collections import namedtuple
from skimage.transform import SimilarityTransform
from sklearn.model_selection import train_test_split
from openem_train.ssd import dataset

FishDetection = namedtuple('FishDetection', ['video_id', 'frame', 'fish_number', 'x1', 'y1', 'x2', 'y2', 'class_id'])
RulerPoints = namedtuple('RulerPoints', ['x1', 'y1', 'x2', 'y2'])


class FishDetectionDataset:
    def __init__(self, config, is_test=False):
        self.is_test = is_test
        self.config = config

        if is_test:
            self.video_clips = dataset.video_clips_test(config)
            self.fn_suffix = '_test'
        else:
            self.video_clips = dataset.video_clips(config)
            self.fn_suffix = ''

        self.detections = self.load()  # type: Dict[FishDetection]
        self.train_clips, self.test_clips = train_test_split(sorted(self.detections.keys()),
                                                             test_size=0.05,
                                                             random_state=12)

        self.nb_train_samples = sum([len(self.detections[clip]) for clip in self.train_clips])
        self.nb_test_samples = sum([len(self.detections[clip]) for clip in self.test_clips])

        self.ruler_points = {}
        if is_test:
            ruler_points = pd.read_csv(config.test_ruler_position())
        else:
            ruler_points = pd.read_csv(config.train_ruler_position())
        for _, row in ruler_points.iterrows():
            self.ruler_points[row.video_id] = RulerPoints(x1=row.ruler_x0, y1=row.ruler_y0, x2=row.ruler_x1, y2=row.ruler_y1)

    def load(self):
        detections = {}
        ds = pd.read_csv(self.config.train_ann_path())

        for row_id, row in ds.iterrows():
            video_id = row.video_id
            if video_id not in detections:
                detections[video_id] = []

            detections[video_id].append(
                FishDetection(
                    video_id=video_id,
                    frame=row.frame,
                    fish_number=row.fish_number,
                    x1=row.x1, y1=row.y1,
                    x2=row.x2, y2=row.y2,
                    class_id=row.species_id
                )
            )
        # load labeled no fish images
        for fn in self.config.no_fish_examples():
            # file name format: video_frame.jpg
            fn = os.path.basename(fn)
            fn = fn[:-len('.jpg')]
            video_id, frame = fn.split('_')
            frame = int(frame)

            if video_id not in detections:
                detections[video_id] = []

            detections[video_id].append(
                FishDetection(
                    video_id=video_id,
                    frame=frame,
                    fish_number=0,
                    x1=np.nan, y1=np.nan,
                    x2=np.nan, y2=np.nan,
                    class_id=0
                )
            )
        return detections

    def transform_for_clip(self, video_id, dst_w=720, dst_h=360, points_random_shift=0):
        img_points = np.array([[dst_w * 0.1, dst_h / 2], [dst_w * 0.9, dst_h / 2]])
        points = self.ruler_points[video_id]

        ruler_points = np.array([[points.x1, points.y1], [points.x2, points.y2]])

        if points_random_shift > 0:
            img_points += np.random.uniform(-points_random_shift, points_random_shift, (2, 2))

        tform = SimilarityTransform()
        tform.estimate(dst=ruler_points, src=img_points)

        return tform
