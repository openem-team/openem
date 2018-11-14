import scipy
import skimage
import random
import numpy as np
from multiprocessing.pool import ThreadPool
from keras.applications.imagenet_utils import preprocess_input
from openem_train.ssd import fish_detection
from openem_train.ssd import dataset
from openem_train.util import utils
from openem_train.util import img_augmentation

class SampleCfg:
    """
    Configuration structure for crop parameters.
    """

    def __init__(self,
                 detection,
                 transformation,
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

    def __lt__(self, other):
        return True

    def __str__(self):
        return dataset.CLASSES[self.detection.class_id] + ' ' + str(self.__dict__)

class SSDDataset(fish_detection.FishDetectionDataset):
    def __init__(self, config, bbox_util, preprocess_input=preprocess_input, is_test=False):
        super().__init__(config, is_test)
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
        img = scipy.misc.imread(dataset.image_fn(self.config, cfg.detection.video_id, cfg.detection.frame, is_test=self.is_test))
        crop = skimage.transform.warp(img, cfg.transformation, mode='edge', order=3, output_shape=(self.config.detect_height(), self.config.detect_width()))

        detection = cfg.detection

        if detection.class_id > 0:
            coords = np.array([[detection.x1, detection.y1], [detection.x2, detection.y2]])
            coords_in_crop = cfg.transformation.inverse(coords)
            aspect_ratio = dataset.ASPECT_RATIO_TABLE[dataset.CLASSES[detection.class_id]]
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

    def generate_x_from_precomputed_crop(self, cfg: SampleCfg):
        crop = scipy.misc.imread(dataset.image_crop_fn(cfg.detection.video_id, cfg.detection.frame, is_test=self.is_test))
        crop = crop.astype('float32')
        # print('crop max val:', np.max(crop))
        return crop

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
                cfg = SampleCfg(detection=detection, transformation=tform)

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

    def generate_x_for_train_video_id(self, video_id, batch_size, pool, frames=None):
        detections = []  # type: List[fish_detection.FishDetection]
        frames_to_use = frames if frames is not None else range(len(dataset.video_clips(is_test=self.is_test)[video_id]))
        for frame_id in frames_to_use:
            detections.append(
                fish_detection.FishDetection(
                    video_id=video_id,
                    frame=frame_id,
                    fish_number=0,
                    x1=np.nan, y1=np.nan,
                    x2=np.nan, y2=np.nan,
                    class_id=0
                )
            )

        def output_samples(samples_to_process):
            inputs = []
            for img in pool.map(self.generate_x_from_precomputed_crop, samples_to_process):
                inputs.append(img)

            frames = [cfg.detection.frame for cfg in samples_to_process]

            tmp_inp = np.array(inputs)
            inputs.clear()  # lets return some memory earlier
            x = self.preprocess_input(tmp_inp)
            return x, frames

        samples_to_process = []
        for detection in detections:
            cfg = SampleCfg(detection=detection, transformation=None)
            samples_to_process.append(cfg)

            if len(samples_to_process) >= batch_size:
                yield output_samples(samples_to_process)
                samples_to_process = []

        if len(samples_to_process) > 0:
            yield output_samples(samples_to_process)

