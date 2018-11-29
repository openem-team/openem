import os
import scipy
import skimage
import random
import numpy as np
import pandas as pd
from typing import List, Dict
from collections import namedtuple
from multiprocessing.pool import ThreadPool
from skimage.transform import SimilarityTransform
from sklearn.model_selection import train_test_split
from keras.applications.imagenet_utils import preprocess_input
from openem_train.util import utils
from openem_train.util import img_augmentation

FishDetection = namedtuple('FishDetection', ['video_id', 'frame', 'fish_number', 'x1', 'y1', 'x2', 'y2', 'class_id'])
RulerPoints = namedtuple('RulerPoints', ['x1', 'y1', 'x2', 'y2'])

class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self,
                 detection,
                 transformation,
                 class_names,
                 saturation=0.5, 
                 contrast=0.5, 
                 brightness=0.5,  # 0.5  - no changes, range 0..1
                 blurred_by_downscaling=1,
                 hflip=False,
                 vflip=False):
        self.transformation = transformation
        self.detection = detection
        self.vflip = vflip
        self.hflip = hflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blurred_by_downscaling = blurred_by_downscaling
        self.class_names = class_names

    def __lt__(self, other):
        return True

    def __str__(self):
        return self.class_names[self.detection.class_id] + ' ' + str(self.__dict__)

class SSDDataset:
    def __init__(self, config, bbox_util, preprocess_input=preprocess_input):
        self.config = config
        self.fn_suffix = ''
        self.detections = self.load()  # type: Dict[FishDetection]
        self.train_clips, self.test_clips = train_test_split(sorted(self.detections.keys()),
                                                             test_size=0.05,
                                                             random_state=12)

        self.nb_train_samples = sum([len(self.detections[clip]) for clip in self.train_clips])
        self.nb_test_samples = sum([len(self.detections[clip]) for clip in self.test_clips])

        self.ruler_points = {}
        ruler_points = pd.read_csv(config.train_ruler_position())
        for _, row in ruler_points.iterrows():
            self.ruler_points[row.video_id] = RulerPoints(x1=row.ruler_x0, y1=row.ruler_y0, x2=row.ruler_x1, y2=row.ruler_y1)

        self.bbox_util = bbox_util
        self.preprocess_input = preprocess_input

    def horizontal_flip(self, img, y):
        img = img[:, ::-1]
        if y.size:
            y[:, [0, 2]] = 1 - y[:, [2, 0]]
        return img, y

    def vertical_flip(self, img, y):
        img = img[::-1]
        if y.size:
            y[:, [1, 3]] = 1 - y[:, [3, 1]]
        return img, y

    def generate_xy(self, cfg: SampleCfg):
        img = scipy.misc.imread(os.path.join(
            self.config.train_imgs_dir(), 
            cfg.detection.video_id, 
            '{:04}.jpg'.format(cfg.detection.frame)))
        crop = skimage.transform.warp(img, cfg.transformation, mode='edge', order=3, output_shape=(self.config.detect_height(), self.config.detect_width()))

        detection = cfg.detection

        if detection.class_id > 0:
            coords = np.array([[detection.x1, detection.y1], [detection.x2, detection.y2]])
            coords_in_crop = cfg.transformation.inverse(coords)
            aspect_ratio = self.config.aspect_ratios()[detection.class_id - 1]
            coords_box0, coords_box1 = utils.bbox_for_line(coords_in_crop[0, :], coords_in_crop[1, :], aspect_ratio)
            coords_box0 /= np.array([self.config.detect_width(), self.config.detect_height()])
            coords_box1 /= np.array([self.config.detect_width(), self.config.detect_height()])
            targets = [coords_box0[0], coords_box0[1], coords_box1[0], coords_box1[1]]


            cls = [0] * (self.config.num_classes() - 1)
            cls[detection.class_id-1] = 1
            targets = np.array([targets+cls])
        else:
            targets = np.array([])

        crop = crop.astype('float32')
        if cfg.saturation != 0.5:
            crop = img_augmentation.saturation(crop, variance=0.25, r=cfg.saturation)

        if cfg.contrast != 0.5:
            crop = img_augmentation.contrast(crop, variance=0.25, r=cfg.contrast)

        if cfg.brightness != 0.5:
            crop = img_augmentation.brightness(crop, variance=0.3, r=cfg.brightness)

        if cfg.hflip:
            crop, targets = self.horizontal_flip(crop, targets)

        if cfg.vflip:
            crop, targets = self.vertical_flip(crop, targets)

        crop = img_augmentation.blurred_by_downscaling(crop, 1.0/cfg.blurred_by_downscaling)

        return crop*255.0, targets

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

    def generate_ssd(self, batch_size, is_training, verbose=False, skip_assign_boxes=False, always_shuffle=False):
        pool = ThreadPool(processes=8)

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        detections = []  # type: List[fish_detection.FishDetection]
        if is_training:
            detections += sum([self.detections[video_id] for video_id in self.train_clips], [])
        else:
            detections += sum([self.detections[video_id] for video_id in self.test_clips], [])

        while True:
            points_random_shift = 0
            samples_to_process = []
            if is_training or always_shuffle:
                random.shuffle(detections)
                points_random_shift = 32

            for detection in detections:
                tform = self.transform_for_clip(
                    detection.video_id,
                    dst_w=self.config.detect_width(), 
                    dst_h=self.config.detect_height(),
                    points_random_shift=points_random_shift)
                cfg = SampleCfg(detection=detection, transformation=tform, class_names=self.config.species())

                if is_training:
                    cfg.contrast = rand_or_05()
                    cfg.brightness = rand_or_05()
                    cfg.saturation = rand_or_05()
                    cfg.hflip = random.choice([True, False])
                    cfg.vflip = random.choice([True, False])
                    cfg.blurred_by_downscaling = np.random.choice([1, 1, 1, 1, 2, 2.5, 3, 4])

                if verbose:
                    print(str(cfg))

                samples_to_process.append(cfg)

                if len(samples_to_process) >= batch_size:
                    inputs = []
                    targets = []
                    for img, y in pool.map(self.generate_xy, samples_to_process):
                    # for img, y in map(self.generate_xy, samples_to_process):
                        inputs.append(img)
                        if skip_assign_boxes:
                            targets.append(y)
                        else:
                            targets.append(self.bbox_util.assign_boxes(y))

                    tmp_inp = np.array(inputs)
                    inputs.clear()  # lets return some memory earlier
                    samples_to_process = []
                    x = self.preprocess_input(tmp_inp)
                    y = np.array(targets)

                    yield x, y

