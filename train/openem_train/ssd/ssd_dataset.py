__copyright__ = "Copyright (C) 2018 CVision AI."
__license__ = "GPLv3"
# This file is part of OpenEM, released under GPLv3.
# OpenEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

"""Classes for interfacing with SSD training data.
"""

import os
import random
from collections import namedtuple
from typing import List, Dict
from multiprocessing.pool import ThreadPool
import scipy
import numpy as np
import pandas as pd
import skimage
from sklearn.model_selection import train_test_split
from keras.applications.imagenet_utils import preprocess_input
from openem_train.util import utils
from openem_train.util import img_augmentation
from openem_train.util.roi_transform import RoiTransform
from openem_train.util.img_augmentation import resizeAndFill
import math

# pylint: disable=too-many-instance-attributes
FishDetection = namedtuple(
    'FishDetection',
    ['video_id', 'frame', 'x1', 'y1', 'x2', 'y2', 'class_id'])
FishBoxDetection = namedtuple(
    'FishBoxDetection',
    ['video_id', 'frame', 'x', 'y', 'width', 'height', 'theta', 'class_id'])

def appendToDetections(detections, row):
    """
    Append the given csv row to the detections for that video
    :param detections_for_video: list or list-like object to append to.
    :param row: Row from the pandas csv reader
    """
    keys = row.keys()
    linekeys=['x1','x2','y1','y2']
    boxkeys=['x','y','width','height','theta']

    if all(x in keys for x in linekeys):
        detections.append(
            FishDetection(
                video_id=row.video_id,
                frame=row.frame,
                x1=row.x1, y1=row.y1,
                x2=row.x2, y2=row.y2,
                class_id=row.species_id
            )
        )
    elif all(x in keys for x in boxkeys):
        detections.append(
            FishBoxDetection(
                video_id=row.video_id,
                frame=row.frame,
                x=row.x, y=row.y,
                width=row.width,
                height=row.height,
                theta=row.theta,
                class_id=row.species_id
            )
        )
    else:
        raise Exception('Unknown row definiton {}'.format(keys))
def _horizontal_flip(img, bbox):
    """Flips image and bounding box horizontally.

    # Arguments
        img: Image to flip.
        bbox: Bounding box coordinates.

    # Returns
        Tuple containing flipped image and bounding box.
    """
    img = img[:, ::-1]
    if bbox.size:
        bbox[:, [0, 2]] = 1 - bbox[:, [2, 0]]
    return img, bbox

def _vertical_flip(img, bbox):
    """Flips image and bounding box vertically.

    # Arguments
        img: Image to flip.
        bbox: Bounding box coordinates.

    # Returns
        Tuple containing flipped image and bounding box.
    """
    img = img[::-1]
    if bbox.size:
        bbox[:, [1, 3]] = 1 - bbox[:, [3, 1]]
    return img, bbox

class SampleCfg:
    """Configuration structure for crop parameters.
    """

    def __init__(self,
                 detection,
                 transformation,
                 class_names):
        self.transformation = transformation
        self.detection = detection
        self.vflip = False
        self.hflip = False
        self.brightness = 0.5
        self.contrast = 0.5
        self.saturation = 0.5
        self.blurred_by_downscaling = 1
        self.class_names = class_names

    def __lt__(self, other):
        return True

    def __str__(self):
        return self.class_names[self.detection.class_id] + ' ' + str(self.__dict__)

class SSDDataset:
    """Class for interfacing with SSD training data.
    """

    def __init__(self, config, bbox_util, preproc=preprocess_input):
        """Constructor.

        # Arguments:
            config: ConfigInterface object.
            bbox_util: BBoxUtility object.
            preproc: Function to be used for preprocessing.
        """
        self.config = config
        self.fn_suffix = ''
        self.detections = self.load()  # type: Dict[FishDetection]
        self.train_clips, self.test_clips = train_test_split(
            sorted(self.detections.keys()),
            test_size=0.05,
            random_state=12)

        self.nb_train_samples = sum(
            [len(self.detections[clip]) for clip in self.train_clips])
        self.nb_test_samples = sum(
            [len(self.detections[clip]) for clip in self.test_clips])

        self.roi_transform = RoiTransform(config)
        self.bbox_util = bbox_util
        self.preprocess_input = preproc
        self.frame_jitter = list(range(1+self.config.detect_frame_jitter()))
        self.frame_jitter += [-a for a in self.frame_jitter]

    def load(self):
        """ Loads data to be used from annotation csv file.
        """
        detections = {}
        length = pd.read_csv(self.config.length_path())

        for _, row in length.iterrows():
            video_id = row.video_id
            if video_id not in detections:
                detections[video_id] = []
            appendToDetections(detections[video_id],row)

        # load labeled no fish images
        cover = pd.read_csv(self.config.cover_path())
        num_no_fish = 0
        for _, row in cover.iterrows():
            if row.cover == 0:
                detections[row.video_id].append(
                    FishDetection(
                        video_id=row.video_id,
                        frame=row.frame,
                        x1=np.nan, y1=np.nan,
                        x2=np.nan, y2=np.nan,
                        class_id=0
                    )
                )
                num_no_fish += 1
        return detections

    def generate_xy(self, cfg: SampleCfg):
        """Loads and augments SSD training examples.

        # Arguments
            cfg: SampleCfg object.
        """
        # Allow a random frame variation, which creates
        # training data for some covered examples. This also assumes
        # the fish is not moving much within the frame jitter window.
        diff = random.choice(self.frame_jitter)
        frame = cfg.detection.frame + diff
        frame = max(0, frame)
        def get_path(frame_num):
            return os.path.join(
                self.config.train_imgs_dir(),
                cfg.detection.video_id,
                '{:04}.jpg'.format(frame_num))
        if not os.path.exists(get_path(frame)):
            frame -= 1
        if os.stat(get_path(frame)).st_size == 0:
            frame -= 1
        img = scipy.misc.imread(get_path(frame))
        # because each frame may have different dimensions resize and
        # fill to match the detection size prior to warping
        img,scale = resizeAndFill(img, (self.config.detect_height(),
                                  self.config.detect_width())
        )

        crop = skimage.transform.warp(
            img,
            cfg.transformation,
            mode='edge',
            order=3,
            output_shape=(
                self.config.detect_height(),
                self.config.detect_width()))

        detection = cfg.detection

        if detection.class_id > 0:
            if type(detection) == FishDetection:
                coords = np.array([
                    [detection.x1*scale[1], detection.y1*scale[0]],
                    [detection.x2*scale[1], detection.y2*scale[0]]])
                aspect_ratio = self.config.aspect_ratios()[detection.class_id - 1]

                coords_in_crop = cfg.transformation.inverse(coords)
                coords_box0, coords_box1 = utils.bbox_for_line(
                    coords_in_crop[0, :],
                    coords_in_crop[1, :],
                    aspect_ratio)

            elif type(detection) == FishBoxDetection:
                # Rotate box to theta
                rotated_coords=utils.rotate_detection(detection)
                rotated_coords[0] = rotated_coords[0]*scale[1]
                rotated_coords[1] = rotated_coords[1]*scale[0]
                rotated_coords[2] = rotated_coords[2]*scale[1]
                rotated_coords[3] = rotated_coords[3]*scale[0]
                # Translate to ruler space
                coords_in_crop = cfg.transformation.inverse(rotated_coords)
                # Use find corners to be safe
                topLeftIdx,bottomRightIdx=utils.find_corners(coords_in_crop)
                # These are now the diagnol representing the bounding box.
                coords_box0=coords_in_crop[topLeftIdx]
                coords_box1=coords_in_crop[bottomRightIdx]

            coords_box0 /= np.array([
                self.config.detect_width(),
                self.config.detect_height()])
            coords_box1 /= np.array([
                self.config.detect_width(),
                self.config.detect_height()])
            targets = [
                coords_box0[0],
                coords_box0[1],
                coords_box1[0],
                coords_box1[1]]

            cls = [0] * (self.config.num_classes() - 1)
            cls[detection.class_id-1] = 1
            targets = np.array([targets+cls])
        else:
            targets = np.array([])

        crop = crop.astype('float32')
        if cfg.saturation != 0.5:
            crop = img_augmentation.saturation(crop, variance=0.25, mean=cfg.saturation)

        if cfg.contrast != 0.5:
            crop = img_augmentation.contrast(crop, variance=0.25, mean=cfg.contrast)

        if cfg.brightness != 0.5:
            crop = img_augmentation.brightness(crop, variance=0.3, mean=cfg.brightness)

        if cfg.hflip:
            crop, targets = _horizontal_flip(crop, targets)

        if cfg.vflip:
            crop, targets = _vertical_flip(crop, targets)

        crop = img_augmentation.blurred_by_downscaling(crop, 1.0/cfg.blurred_by_downscaling)

        return crop*255.0, targets

    def get_config(self, detection, is_training):
        """Gets sample config for training or validation.

        # Arguments
            detection: Detection to augment.
            is_training: True if training, False if validating.

        # Returns
            SampleCfg object.
        """
        points_random_shift = 0
        if is_training:
            points_random_shift = 32
        tform = self.roi_transform.transform_for_clip(
            detection.video_id,
            dst_w=self.config.detect_width(),
            dst_h=self.config.detect_height(),
            points_random_shift=points_random_shift)
        cfg = SampleCfg(
            detection=detection,
            transformation=tform,
            class_names=self.config.species())

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        if is_training:
            cfg.contrast = rand_or_05()
            cfg.brightness = rand_or_05()
            cfg.saturation = rand_or_05()
            cfg.hflip = random.choice([True, False])
            cfg.vflip = random.choice([True, False])
            cfg.blurred_by_downscaling = np.random.choice(
                [1, 1, 1, 1, 2, 2.5, 3, 4])
        return cfg

    def generate_ssd(self, batch_size, is_training):
        """Generator for SSD training examples.

        # Arguments:
            batch_size: Size of the batch.
            is_training: True for training, False for validating.
        """
        pool = ThreadPool(processes=8)

        detections = []  # type: List[fish_detection.FishDetection]
        if is_training:
            detections += sum([self.detections[video_id] for video_id in self.train_clips], [])
        else:
            detections += sum([self.detections[video_id] for video_id in self.test_clips], [])

        while True:
            samples_to_process = []
            if is_training:
                random.shuffle(detections)

            for detection in detections:

                cfg = self.get_config(detection, is_training)
                samples_to_process.append(cfg)

                if len(samples_to_process) >= batch_size:
                    inputs = []
                    targets = []
                    for img, bbox in pool.map(self.generate_xy, samples_to_process):
                        inputs.append(img)
                        targets.append(self.bbox_util.assign_boxes(bbox))

                    tmp_inp = np.array(inputs)
                    inputs.clear()  # lets return some memory earlier
                    samples_to_process = []

                    yield self.preprocess_input(tmp_inp), np.array(targets)
