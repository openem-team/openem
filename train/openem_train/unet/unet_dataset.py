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
from copy import copy
from multiprocessing.pool import ThreadPool
import numpy as np
import scipy.misc
from openem_train.util.utils import chunks

def preprocess_input(img):
    return img.astype(np.float32) / 128.0 - 1.0

class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self, img_idx,
                 scale_rect_x=1.0,
                 scale_rect_y=1.0,
                 shift_x_ratio=0.0,
                 shift_y_ratio=0.0,
                 angle=0.0,
                 saturation=0.5, contrast=0.5, brightness=0.5,  # 0.5  - no changes, range 0..1
                 hflip=False,
                 vflip=False):
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
        self.blurred_by_downscaling = None

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
        image_files = glob.glob(os.path.join(config.train_mask_imgs_dir(), '*'))

        # Try to find corresponding masks.
        mask_files = []
        for f in image_files:
            path, fname = os.path.split(f)
            base, ext = os.path.splitext(fname)
            patt = os.path.join(config.train_mask_masks_dir(), base + '.*')
            mask = glob.glob(patt)
            if len(mask) != 1:
                msg = "Could not find mask image corresponding to {}!"
                msg += " Searched at: {}"
                raise ValueError(msg.format(f, patt))
            mask_files += mask

        # Load in the masks and images.
        self.images, self.masks = self.load(image_files, mask_files)
        all_idx = list(range(self.images.shape[0]))
        self.train_idx = all_idx[:-96]
        self.test_idx = all_idx[-96:]

    def load(self, image_files, mask_files):
        def load_image(img_fn):
            img_data = scipy.misc.imread(img_fn)
            img_data = scipy.misc.imresize(img_data, 0.5, interp='cubic')
            return img_data

        def load_mask(mask_fn):
            mask_data = scipy.misc.imread(mask_fn, mode='L')
            mask_data = scipy.misc.imresize(mask_data, 0.5, interp='bilinear', mode='L')
            return mask_data

        pool = ThreadPool(processes=8)
        images = pool.map(load_image, image_files)
        images = np.array(images)
        masks = pool.map(load_mask, mask_files)
        masks = np.array(masks)
        return images, masks

    def prepare_x(self, cfg: SampleCfg):
        img = preprocess_input(self.images[cfg.img_idx])
        return img

    def prepare_y(self, cfg: SampleCfg):
        return np.expand_dims(self.masks[cfg.img_idx].astype(np.float32) / 256.0, axis=3)

    def generate(self, batch_size):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]


        while True:
            img_idx = random.choice(self.train_idx)
            cfg = SampleCfg(img_idx=img_idx)
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
                samples_to_process = [SampleCfg(idx) for idx in idxs]
                X_batch = np.array(pool.map(self.prepare_x, samples_to_process))
                y_batch = np.array(pool.map(self.prepare_y, samples_to_process))
                yield X_batch, y_batch


