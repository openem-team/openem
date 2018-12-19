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
            saturation=0.5, contrast=0.5, brightness=0.5, color_shift=0.5,  # 0.5  - no changes, range 0..1
            scale_rect_x=1.0, scale_rect_y=1.0,
            shift_x_ratio=0.0, shift_y_ratio=0.0,
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
        y = np.clip(fish_classification.y, config.detect_height() / 2 - 64, config.detect_height() / 2 + 64)

        if random_pos or (fish_classification.cover_class == CLASS_NO_FISH_ID and abs(x) < 0.01):
            w = random.randrange(200, 360)
            x = random.randrange(200, config.detect_width() - 200)
            y = random.randrange(config.detect_height() / 2 - 64, config.detect_height() / 2 + 64)

        self.rect = Rect(x=x - w / 2, y=y - w / 2, w=w, h=w)

    def __lt__(self, other):
        return True

    def __str__(self):
        return dataset.CLASSES[self.fish_classification.species_class] + ' ' + str(self.__dict__)

def guess_species(known_species, frame_id):
    known_frames = sorted(known_species.keys())
    if len(known_frames) == 0:
        return None

    for i, frame in enumerate(known_frames):
        if frame == frame_id:
            return known_species[frame]
        elif frame > frame_id:
            if i == 0:
                return known_species[frame]
            if known_species[frame] == known_species[known_frames[i - 1]]:
                return known_species[frame]
            else:
                return None

    return known_species[known_frames[-1]]

def get_all_video_ids(config):
    video_ids = []
    for vid in config.train_vids():
        path, f = os.path.split(vid)
        vid_id, _ = os.path.splitext(f)
        if vid_id not in video_ids:
            video_ids.append(vid_id)
    return video_ids

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

        all_video_ids = get_all_video_ids(config)
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
        return int(len(self.train_data) / 2 // batch_size)

    def test_batches(self, batch_size):
        return int(len(self.test_data) // batch_size)

    def load(self):
        """ Loads data to be used from annotation csv file.
        """
        length = pd.read_csv(self.config.length_path())
        cover = pd.read_csv(self.config.cover_path())
        detections = pd.read_csv(self.config.detect_inference_path())
        data = []

        # Load in length data.
        known_species = {}
        for _, row in length.iterrows():
            if row['species_id'] == 0:
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
                dets = detections.loc[(
                    (detections['video_id'] == row['video_id']) &
                    (detections['frame'] == row['frame'])
                )]
                if not dets.empty:
                    det = dets.iloc[0] # Only use the first detection
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
                dets = detections.loc[(
                    (detections['video_id'] == row['video_id']) &
                    (detections['frame'] == row['frame'])
                )]
                if not dets.empty:
                    det = dets.iloc[0] # Only use the first detection
                    species_class = guess_species(
                        known_species[row['video_id']],
                        row['frame'])
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
        img = scipy.misc.imread(
            self.config.train_roi_img(
                cfg.fish_classification.video_id,
                cfg.fish_classification.frame
            )
        )

        crop = utils.get_image_crop(full_rgb=img, rect=cfg.rect,
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
            crop = img_augmentation.blurred_by_downscaling(crop, 1.0 / cfg.blurred_by_downscaling)
        return crop * 255.0

    def generate_xy(self, cfg: SampleCfg):
        return self.generate_x(cfg), cfg.fish_classification.species_class, cfg.fish_classification.cover_class

    def generate(self, batch_size, skip_pp=False, verbose=False):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        def rand_or_05():
            if random.random() > 0.5:
                return random.random()
            return 0.5

        while True:
            sample = random.choice(self.train_data)  # type: FishClassification
            cfg = SampleCfg(self.config,
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
                # batch_samples = [self.generate_xy(sample) for sample in samples_to_process]
                X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
                y_batch_species = np.array([batch_sample[1] for batch_sample in batch_samples])
                y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
                if not skip_pp:
                    X_batch = preprocess_input(X_batch)
                    y_batch_species = to_categorical(y_batch_species, num_classes=self.config.num_classes())
                    y_batch_cover = to_categorical(y_batch_cover, num_classes=3)
                samples_to_process = []
                yield X_batch, {'cat_species': y_batch_species, 'cat_cover': y_batch_cover}

    def generate_test(self, batch_size, skip_pp=False, verbose=False):
        pool = ThreadPool(processes=8)
        samples_to_process = []  # type: [SampleCfg]

        while True:
            for sample in self.test_data[:int(len(self.test_data) // batch_size) * batch_size]:
                cfg = SampleCfg(self.config, fish_classification=sample)
                samples_to_process.append(cfg)

                if len(samples_to_process) == batch_size:
                    batch_samples = pool.map(self.generate_xy, samples_to_process)
                    X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
                    y_batch_species = np.array([batch_sample[1] for batch_sample in batch_samples])
                    y_batch_cover = np.array([batch_sample[2] for batch_sample in batch_samples])
                    if not skip_pp:
                        X_batch = preprocess_input(X_batch)
                        y_batch_species = to_categorical(y_batch_species, num_classes=self.config.num_classes())
                        y_batch_cover = to_categorical(y_batch_cover, num_classes=3)
                    samples_to_process = []
                    yield X_batch, {'cat_species': y_batch_species, 'cat_cover': y_batch_cover}

    def generate_full_test_for_clip(self, batch_size, pool, video_id, skip_pp=False):
        all_configs = [SampleCfg(self.config, fish_classification=sample) for sample in self.test_data_for_clip[video_id]]
        all_configs = sorted(all_configs, key=lambda x: x.frame)
        for samples_to_process in utils.chunks(all_configs, batch_size):
            batch_samples = pool.map(self.generate_xy, samples_to_process)
            X_batch = np.array([batch_sample[0] for batch_sample in batch_samples])
            if not skip_pp:
                X_batch = preprocess_input(X_batch)
            yield X_batch



