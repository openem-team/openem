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

"""Classes for defining UNet dataset."""

import glob
import os
import random
from collections import namedtuple
from copy import copy
from multiprocessing.pool import ThreadPool
import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from openem_train.util.utils import chunks
from openem_train.util import img_augmentation
from openem_train.util import utils

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])

class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self, img_idx, config,
                 scale_rect_x=1.0,
                 scale_rect_y=1.0,
                 shift_x_ratio=0.0,
                 shift_y_ratio=0.0,
                 angle=0.0,
                 saturation=0.5, contrast=0.5, brightness=0.5,  # 0.5  - no changes, range 0..1
                 hflip=False,
                 vflip=False,
                 blurred_by_downscaling=1):
        self.angle = angle
        self.shift_y_ratio = shift_y_ratio
        self.shift_x_ratio = shift_x_ratio
        self.scale_rect_y = scale_rect_y
        self.scale_rect_x = scale_rect_x
        self.img_idx = img_idx
        self.vflip = vflip
        self.hflip = hflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blurred_by_downscaling = blurred_by_downscaling

        orig_w = config.find_ruler_width()
        orig_h = config.find_ruler_height()
        w = random.randrange(int(0.5 * orig_w), orig_w)
        x = random.randrange(0, orig_w - w)
        h = int(w * orig_h / orig_w)
        y = random.randrange(0, orig_h - h)
        self.rect = Rect(x=x, y=y, w=w, h=h)

    def __lt__(self, other):
        return True

    def __str__(self):
        dc = copy(self.__dict__)
        del dc['img']
        return str(dc)

class UnetDataset:
    def __init__(self, config):
        self.config = config

        # Find all images.
        self.image_files = glob.glob(os.path.join(config.train_mask_imgs_dir(), '*'))

        # Build list of corresponding mask files.
        self.mask_files = [
            os.path.join(
                config.train_mask_masks_dir(),
                os.path.basename(fname),
            ) for fname in self.image_files
        ]

        # Load in the masks and images.
        all_idx = list(range(len(self.image_files)))
        random.shuffle(all_idx)
        split_idx = int(len(all_idx) * 0.9)
        self.train_idx = all_idx[:-split_idx]
        self.test_idx = all_idx[-split_idx:]

        # Store number of channels
        self.num_channels = self.config.find_ruler_num_channels()

    def load_image(self, img_fn):
        img_data = imread(img_fn)[:, :, :self.num_channels]
        img_data = rescale(img_data, 0.5)
        return img_data

    def load_mask(self, mask_fn):
        mask_data = imread(mask_fn, as_gray=True)
        mask_data = rescale(mask_data, 0.5)
        return mask_data

    def prepare_x(self, cfg: SampleCfg):
        img = self.load_image(self.image_files[cfg.img_idx])
        crop = utils.get_image_crop(
            full_rgb=img, rect=cfg.rect,
            scale_rect_x=cfg.scale_rect_x, scale_rect_y=cfg.scale_rect_y,
            shift_x_ratio=cfg.shift_x_ratio, shift_y_ratio=cfg.shift_y_ratio,
            angle=cfg.angle,
            out_size=(self.config.find_ruler_width(), self.config.find_ruler_height()),
            square=False)

        crop = crop.astype('float32')
        if cfg.saturation != 0.5:
            crop = img_augmentation.saturation(crop, variance=0.2, mean=cfg.saturation)

        if cfg.contrast != 0.5:
            crop = img_augmentation.contrast(crop, variance=0.25, mean=cfg.contrast)

        if cfg.brightness != 0.5:
            crop = img_augmentation.brightness(crop, variance=0.3, mean=cfg.brightness)

        if cfg.hflip:
            crop = img_augmentation.horizontal_flip(crop)

        if cfg.vflip:
            crop = img_augmentation.vertical_flip(crop)

        if cfg.blurred_by_downscaling != 1:
            crop = img_augmentation.blurred_by_downscaling(
                crop,
                1.0 / cfg.blurred_by_downscaling
            )
        # Center the input to interval -1.0, 1.0
        crop = 2.0 * crop - 1.0
        return crop

    def prepare_y(self, cfg: SampleCfg):
        img = self.load_mask(self.mask_files[cfg.img_idx])
        img = np.expand_dims(img.astype(np.float32), axis=3)
        crop = utils.get_image_crop(
            full_rgb=img, rect=cfg.rect,
            scale_rect_x=cfg.scale_rect_x, scale_rect_y=cfg.scale_rect_y,
            shift_x_ratio=cfg.shift_x_ratio, shift_y_ratio=cfg.shift_y_ratio,
            angle=cfg.angle, square=False,
            out_size=(self.config.find_ruler_width(), self.config.find_ruler_height()))

        if cfg.hflip:
            crop = img_augmentation.horizontal_flip(crop)

        if cfg.vflip:
            crop = img_augmentation.vertical_flip(crop)

        return crop

    def generate(self, batch_size):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        while True:
            img_idx = random.choice(self.train_idx)
            cfg = SampleCfg(
                img_idx=img_idx,
                config=self.config,
                #saturation=rand_or_05(),
                #contrast=rand_or_05(),
                #brightness=rand_or_05(),
                scale_rect_x=random.uniform(0.8, 1.0),
                scale_rect_y=random.uniform(0.8, 1.0),
                shift_x_ratio=random.uniform(-0.1, 0.1),
                shift_y_ratio=random.uniform(-0.1, 0.1),
                angle=random.uniform(-10.0, 10.0),
                hflip=random.choice([True, False]),
                vflip=random.choice([True, False]),
            )
            samples_to_process.append(cfg)

            if len(samples_to_process) == batch_size:
                X_batch = np.array(pool.map(self.prepare_x, samples_to_process))
                y_batch = np.array(pool.map(self.prepare_y, samples_to_process))
                samples_to_process = []
                yield X_batch, y_batch

    def generate_validation(self, batch_size):
        pool = ThreadPool(processes=8)
        while True:
            for idxs in chunks(self.test_idx, batch_size):
                samples_to_process = [SampleCfg(idx, self.config) for idx in idxs]
                X_batch = np.array(pool.map(self.prepare_x, samples_to_process))
                y_batch = np.array(pool.map(self.prepare_y, samples_to_process))
                yield X_batch, y_batch


