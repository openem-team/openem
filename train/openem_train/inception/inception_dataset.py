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

"""Dataset interface for inception.
"""

import os
import random
from collections import namedtuple
from multiprocessing.pool import ThreadPool
import pandas as pd
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split
from keras.applications.inception_v3 import preprocess_input
from keras.utils import to_categorical
from openem_train.util import img_augmentation
from openem_train.util import utils

CLASS_NO_FISH_ID = 0
CLASS_HAND_OVER_ID = 1
CLASS_FISH_CLEAR_ID = 2

FishClassification = namedtuple(
    'FishClassification', [
        'video_id',
        'frame',
        'x', 'y', 'w',
        'species_class',
        'cover_class'])


SSDDetection = namedtuple(
    'SSDDetection', [
        'video_id',
        'frame',
        'x', 'y', 'w', 'h',
        'class_id',
        'confidence'
    ]
)

Rect = namedtuple('Rect', ['x', 'y', 'w', 'h'])

class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(
            self,
            config,
            fish_classification: FishClassification,
            saturation=0.5,
            contrast=0.5,
            brightness=0.5,
            color_shift=0.5,  # 0.5  - no changes, range 0..1
            scale_rect_x=1.0,
            scale_rect_y=1.0,
            shift_x_ratio=0.0,
            shift_y_ratio=0.0,
            angle=0.0,
            hflip=False,
            vflip=False,
            blurred_by_downscaling=1,
            random_pos=False,
            ssd_detection=None):
        self.color_shift = color_shift
        self.ssd_detection = ssd_detection
        self.angle = angle
        self.shift_x_ratio = shift_x_ratio
        self.shift_y_ratio = shift_y_ratio
        self.scale_rect_y = scale_rect_y
        self.scale_rect_x = scale_rect_x
        self.fish_classification = fish_classification
        self.vflip = vflip
        self.hflip = hflip
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.blurred_by_downscaling = blurred_by_downscaling
        self.cache_img = False

        w = np.clip(fish_classification.w + 64, 200, 360)
        x = fish_classification.x
        y = np.clip(
            fish_classification.y,
            config.detect_height() / 2 - 64,
            config.detect_height() / 2 + 64
        )

        if random_pos or (fish_classification.cover_class == CLASS_NO_FISH_ID and abs(x) < 0.01):
            w = random.randrange(200, 360)
            x = random.randrange(200, config.detect_width() - 200)
            y = random.randrange(config.detect_height() / 2 - 64, config.detect_height() / 2 + 64)

        self.rect = Rect(x=x - w / 2, y=y - w / 2, w=w, h=w)

    def __lt__(self, other):
        return True

def guess_species(known_species, frame_id):
    """Returns a guess at a species based frames of known species and the
       frame with a detection having unknown species.

    # Arguments
        known_species: Dict mapping frame to species names.
        frame_id: Frame number of unknown species.

    # Returns
        Name of species.
    """
    known_frames = sorted(known_species.keys())
    if not known_frames:
        return None

    for i, frame in enumerate(known_frames):
        if frame == frame_id:
            return known_species[frame]
        if frame > frame_id:
            if i == 0:
                return known_species[frame]
            if known_species[frame] == known_species[known_frames[i - 1]]:
                return known_species[frame]
            return None

    return known_species[known_frames[-1]]

class InceptionDataset:
    """Dataset interface for inception.
    """
    def __init__(self, config):
        """Constructor.

        # Arguments
            config: ConfigInterface object.
        """
        self.config = config
        self.data = []  # type: List[FishClassification]
        # video_id->frame->species:
        self.known_species = {}  # type: Dict[str, Dict[int, int]]

        all_video_ids = config.all_video_ids()
        self.train_video_ids, self.test_video_ids = train_test_split(
            sorted(all_video_ids),
            test_size=0.05,
            random_state=12)
        self.data, self.known_species = self.load()

        self.train_data = [d for d in self.data if d.video_id in self.train_video_ids]
        self.test_data_full = [d for d in self.data if d.video_id in self.test_video_ids]
        self.test_data = self.test_data_full[::2]

        self.test_data_for_clip = {}
        for d in self.test_data_full:
            if not d.video_id in self.test_data_for_clip:
                self.test_data_for_clip[d.video_id] = []
            self.test_data_for_clip[d.video_id].append(d)

        self.crops_cache = {}

        print('train samples: {} test samples {}'.format(len(self.train_data), len(self.test_data)))

    def train_batches(self, batch_size):
        """Returns number of training batches.
        """
        return int(len(self.train_data) / 2 // batch_size)

    def test_batches(self, batch_size):
        """Returns number of validation batches.
        """
        return int(len(self.test_data) // batch_size)

    def load(self):
        """ Loads data to be used from annotation csv file.
        """
        # This dict is used to multiply the number of samples in
        # each cover class. It is a rough way of class balancing.
        repeat_samples = {
            CLASS_FISH_CLEAR_ID: 1,
            CLASS_HAND_OVER_ID: 4,
            CLASS_NO_FISH_ID: 2
        }
        length = pd.read_csv(self.config.length_path())
        cover = pd.read_csv(self.config.cover_path())
        detections = pd.read_csv(self.config.detect_inference_path())
        data = []

        # Load in length data.
        known_species = {}
        for _, row in length.iterrows():
            if row['species_id'] == 0:
                for _ in range(repeat_samples[CLASS_NO_FISH_ID]):
                    data.append(
                        FishClassification(
                            video_id=row['video_id'],
                            frame=row['frame'],
                            x=0,
                            y=0,
                            w=0,
                            species_class=0,
                            cover_class=CLASS_NO_FISH_ID
                        )
                    )
            else:
                if row['video_id'] not in known_species:
                    known_species[row['video_id']] = {}
                known_species[row['video_id']][row['frame']] = row['species_id']
                det = utils.get_best_detection(
                    row['video_id'],
                    row['frame'],
                    detections
                )
                if not det is None:
                    for _ in range(repeat_samples[CLASS_FISH_CLEAR_ID]):
                        data.append(
                            FishClassification(
                                video_id=det['video_id'],
                                frame=det['frame'],
                                x=det['x'],
                                y=det['y'],
                                w=det['w'],
                                species_class=row['species_id'],
                                cover_class=CLASS_FISH_CLEAR_ID
                            )
                        )

        # Load in cover data.
        for _, row in cover.iterrows():
            if row['cover'] == CLASS_NO_FISH_ID:
                for _ in range(repeat_samples[CLASS_NO_FISH_ID]):
                    data.append(
                        FishClassification(
                            video_id=row['video_id'],
                            frame=row['frame'],
                            x=0,
                            y=0,
                            w=0,
                            species_class=0,
                            cover_class=CLASS_NO_FISH_ID
                        )
                    )
            else:
                det = utils.get_best_detection(
                    row['video_id'],
                    row['frame'],
                    detections
                )
                if not det is None:
                    species_class = guess_species(
                        known_species[row['video_id']],
                        row['frame'])
                    for _ in range(repeat_samples[row['cover']]):
                        data.append(
                            FishClassification(
                                video_id=det['video_id'],
                                frame=det['frame'],
                                x=det['x'],
                                y=det['y'],
                                w=det['w'],
                                species_class=species_class,
                                cover_class=row['cover']
                            )
                        )
        return data, known_species

    def generate_x(self, cfg: SampleCfg):
        """Generates a cropped image, randomized by several parameters.

        # Arguments
            cfg: Configuration for sample randomization.

        # Returns
            Randomized crop.
        """
        img = scipy.misc.imread(
            self.config.train_roi_img(
                cfg.fish_classification.video_id,
                cfg.fish_classification.frame
            )
        )

        crop = utils.get_image_crop(
            full_rgb=img, rect=cfg.rect,
            scale_rect_x=cfg.scale_rect_x, scale_rect_y=cfg.scale_rect_y,
            shift_x_ratio=cfg.shift_x_ratio, shift_y_ratio=cfg.shift_y_ratio,
            angle=cfg.angle, out_size=self.config.classify_height())

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
        return crop * 255.0

    def generate_xy(self, cfg: SampleCfg):
        """Returns randomized crop along with species and cover labels.

        # Arguments
            cfg: Configuration for sample randomization.

        # Returns
            Tuple containing randomized crop with species and cover labels.
        """
        return (
            self.generate_x(cfg),
            cfg.fish_classification.species_class,
            cfg.fish_classification.cover_class
        )

    def generate(self, batch_size, skip_pp=False):
        """Generator function for inception training data.

        # Arguments
            batch_size: Batch size.
            skip_pp: Boolean indicating whether to skip preprocessing.
        """
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        while True:
            sample = random.choice(self.train_data)  # type: FishClassification
            cfg = SampleCfg(
                self.config,
                fish_classification=sample,
                saturation=rand_or_05(),
                contrast=rand_or_05(),
                brightness=rand_or_05(),
                color_shift=rand_or_05(),
                shift_x_ratio=random.uniform(-0.2, 0.2),
                shift_y_ratio=random.uniform(-0.2, 0.2),
                angle=random.uniform(-20.0, 20.0),
                hflip=random.choice([True, False]),
                vflip=random.choice([True, False]),
                blurred_by_downscaling=np.random.choice([1, 1, 1, 1, 1, 1, 1, 1, 2, 2.5, 3, 4])
            )
            samples_to_process.append(cfg)

            if len(samples_to_process) == batch_size:
                batch_samples = pool.map(self.generate_xy, samples_to_process)
                x_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
                y_batch_species = np.array([batch_sample[1] 
                    if batch_sample[1] is not None else 0 for batch_sample in batch_samples])
                y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
                if not skip_pp:
                    x_batch = preprocess_input(x_batch)
                    y_batch_species = to_categorical(
                        y_batch_species,
                        num_classes=self.config.num_classes()
                    )
                    y_batch_cover = to_categorical(
                        y_batch_cover,
                        num_classes=3
                    )
                samples_to_process = []
                yield x_batch, {'cat_species': y_batch_species, 'cat_cover': y_batch_cover}

    def generate_test(self, batch_size, skip_pp=False):
        """Generator function for inception validation data.

        # Arguments
            batch_size: Batch size.
            skip_pp: Boolean indicating whether to skip preprocessing.
        """
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        while True:
            for sample in self.test_data[:int(len(self.test_data) // batch_size) * batch_size]:
                cfg = SampleCfg(self.config, fish_classification=sample)
                samples_to_process.append(cfg)

                if len(samples_to_process) == batch_size:
                    batch_samples = pool.map(self.generate_xy, samples_to_process)
                    x_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
                    y_batch_species = np.array([batch_sample[1] 
                        if batch_sample[1] is not None else 0 for batch_sample in batch_samples])
                    y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
                    if not skip_pp:
                        x_batch = preprocess_input(x_batch)
                        y_batch_species = to_categorical(
                            y_batch_species,
                            num_classes=self.config.num_classes()
                        )
                        y_batch_cover = to_categorical(
                            y_batch_cover,
                            num_classes=3
                        )
                    samples_to_process = []
                    yield x_batch, {'cat_species': y_batch_species, 'cat_cover': y_batch_cover}
