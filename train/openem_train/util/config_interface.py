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

"""Defines interface to config file."""

import os
import glob
import configparser
from datetime import datetime

class ConfigInterface:
    """Interface to config file.
    """
    def __init__(self, config_file):
        """Constructor.

        # Arguments
            config_file: Path to config file.
        """
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        # Read in species info.
        self._species = self.config.get('Data', 'Species').split(',')
        if self.config.has_option('Data', 'LengthFormat') == False:
            self._ratios = self.config.get('Data', 'AspectRatios').split(',')
            self._ratios = [float(r) for r in self._ratios]
            if len(self._ratios) != len(self._species):
                msg = (
                    "Invalid config file!  "
                    "Number of species and aspect ratios must match!  "
                    "Number of species: {}, "
                    "Number of aspect ratios: {}")
                msg.format(len(self._species), len(self._ratios))
                raise ValueError(msg)
        self._num_classes = len(self._species) + 1

    def model_dir(self):
        """Gets model directory.
        """
        return os.path.join(self.config.get('Paths', 'ModelDir'), 'deploy')

    def find_ruler_model_dir(self):
        """Gets find ruler model directory.
        """
        return os.path.join(self.model_dir(), 'find_ruler')

    def find_ruler_model_path(self):
        """Gets find ruler model file path.
        """
        return os.path.join(self.find_ruler_model_dir(), 'find_ruler.pb')

    def detect_model_dir(self):
        """Gets detection model directory.
        """
        return os.path.join(self.model_dir(), 'detect')

    def detect_model_path(self):
        """Gets detection file path.
        """
        return os.path.join(self.detect_model_dir(), 'detect.pb')

    def detect_retinanet_path(self):
        """Gets detection file path.
        """
        return os.path.join(self.detect_model_dir(), 'detect_retinanet.pb')

    def classify_model_dir(self):
        """Gets classification model directory.
        """
        return os.path.join(self.model_dir(), 'classify')

    def classify_model_path(self):
        """Gets classification file path.
        """
        return os.path.join(self.classify_model_dir(), 'classify.pb')

    def count_model_dir(self):
        """Gets count model directory.
        """
        return os.path.join(self.model_dir(), 'count')

    def count_model_path(self):
        """Gets count file path.
        """
        return os.path.join(self.count_model_dir(), 'count.pb')

    def work_dir(self):
        """Gets working directory.
        """
        return self.config.get('Paths', 'WorkDir')

    def train_dir(self):
        """Returns training directory.
        """
        return self.config.get('Paths', 'TrainDir')

    def test_dir(self):
        """Returns test directory.
        """
        return self.config.get('Paths', 'TestDir')

    def num_classes(self):
        """Returns number of classes, including null class.
        """
        return self._num_classes

    def species(self):
        """Returns list of species names.
        """
        return self._species

    def aspect_ratios(self):
        """Returns list of species aspect ratios.
        """
        return self._ratios

    def find_ruler_width(self):
        """Returns width of image used for finding ruler.
        """
        return self.config.getint('FindRuler', 'Width')

    def find_ruler_height(self):
        """Returns height of image used for finding ruler.
        """
        return self.config.getint('FindRuler', 'Height')

    def find_ruler_batch_size(self):
        """Returns batch size used for find ruler training.
        """
        return self.config.getint('FindRuler', 'BatchSize')

    def find_ruler_val_batch_size(self):
        """Returns batch size used for find ruler validation.
        """
        return self.config.getint('FindRuler', 'ValBatchSize')

    def find_ruler_num_epochs(self):
        """Returns number of epochs for find ruler training.
        """
        return self.config.getint('FindRuler', 'NumEpochs')

    def find_ruler_initial_epoch(self):
        """Returns initial epoch for find ruler training.
        """
        return self.config.getint('FindRuler', 'InitialEpoch')

    def detect_width(self):
        """Returns width of ROI used for detection.
        """
        return self.config.getint('Detect', 'Width')

    def detect_height(self):
        """Returns height of ROI used for detection.
        """
        return self.config.getint('Detect', 'Height')

    def detect_batch_size(self):
        """Returns batch size used for detection training.
        """
        return self.config.getint('Detect', 'BatchSize')

    def detect_val_batch_size(self):
        """Returns batch size used for detection validation.
        """
        return self.config.getint('Detect', 'ValBatchSize')

    def detect_val_random_seed(self):
        """ Returns the value to use for the random seed
            to generate the validation population size.
        """
        return self.config.getint('Detect', 'ValRandomSeed')
    def detect_val_population(self):
        """ Returns the population percentage for the validation set """
        return self.config.getfloat('Detect', 'ValPopulation')
    def detect_backbone(self):
        """ Returns the backbone for retinanet to use """
        if self.config.has_option('Detect', 'Backbone'):
            return self.config.get('Detect', 'Backbone')
        else:
            return None
    def detect_force_aspect(self):
        """ Returns the backbone for retinanet to use """
        if self.config.has_option('Detect', 'ForceAspect'):
            return self.config.getfloat('Detect', 'ForceAspect')
        else:
            return None
    def detect_num_epochs(self):
        """Returns number of epochs used for detection training.
        """
        return self.config.getint('Detect', 'NumEpochs')

    def detect_frame_jitter(self):
        """Returns frame jitter setting for detection training.
        """
        return self.config.getint('Detect', 'FrameJitter')

    def detect_initial_epoch(self):
        """Returns initial epoch for detection training.
        """
        return self.config.getint('Detect', 'InitialEpoch')

    def detect_steps_per_epoch(self):
        """Returns steps per epoch for detection traiing if the key exists,
           otherwise returns None.
        """
        steps_per_epoch = None
        if self.config.has_option('Detect', 'StepsPerEpoch'):
            steps_per_epoch = self.config.getint('Detect', 'StepsPerEpoch')
        return steps_per_epoch

    def detect_do_validation(self):
        """Returns whether to do validation if the key exists, otherwise
           returns default value of True.
        """
        do_validation = True
        if self.config.has_option('Detect', 'DoValidation'):
            do_validation = self.config.getboolean('Detect', 'DoValidation')
        return do_validation

    def classify_width(self):
        """Returns width of detections used for classification training.
        """
        return self.config.getint('Classify', 'Width')

    def classify_height(self):
        """Returns height of detections used for classification training.
        """
        return self.config.getint('Classify', 'Height')

    def classify_batch_size(self):
        """Returns batch size used for classification training.
        """
        return self.config.getint('Classify', 'BatchSize')

    def classify_val_batch_size(self):
        """Returns batch size used for classification validation.
        """
        return self.config.getint('Classify', 'ValBatchSize')

    def classify_num_epochs(self):
        """Returns number of epochs used for classification training.
        """
        return self.config.getint('Classify', 'NumEpochs')

    def classify_initial_epoch(self):
        """Returns initial epoch for classify training.
        """
        return self.config.getint('Classify', 'InitialEpoch')

    def classify_steps_per_epoch(self):
        """Returns steps per epoch for classication training if the key exists,
           otherwise returns None.
        """
        steps_per_epoch = None
        if self.config.has_option('Classify', 'StepsPerEpoch'):
            steps_per_epoch = self.config.getint('Classify', 'StepsPerEpoch')
        return steps_per_epoch

    def classify_do_validation(self):
        """Returns whether to do validation if the key exists, otherwise
           returns default value of True.
        """
        do_validation = True
        if self.config.has_option('Classify', 'DoValidation'):
            do_validation = self.config.getboolean('Classify', 'DoValidation')
        return do_validation

    def count_num_steps(self):
        """Returns number of timesteps used as input to count model.
        """
        return self.config.getint('Count', 'NumSteps')

    def count_num_steps_crop(self):
        """Returns number of timesteps to crop for count model.
        """
        return self.config.getint('Count', 'NumStepsCrop')

    def count_num_features(self):
        """Returns number of features used as input to count model.
        """
        return self.config.getint('Count', 'NumFeatures')

    def count_batch_size(self):
        """Returns batch size used for count training.
        """
        return self.config.getint('Count', 'BatchSize')

    def count_val_batch_size(self):
        """Returns batch size used for count validation.
        """
        return self.config.getint('Count', 'ValBatchSize')

    def count_num_epochs(self):
        """Returns number of epochs used for count training.
        """
        return self.config.getint('Count', 'NumEpochs')

    def count_initial_epoch(self):
        """Returns initial epoch for count training.
        """
        return self.config.getint('Count', 'InitialEpoch')

    def count_steps_per_epoch(self):
        """Returns steps per epoch for count traiing if the key exists,
           otherwise returns None.
        """
        steps_per_epoch = None
        if self.config.has_option('Count', 'StepsPerEpoch'):
            steps_per_epoch = self.config.getint('Count', 'StepsPerEpoch')
        return steps_per_epoch

    def count_do_validation(self):
        """Returns whether to do validation if the key exists, otherwise
           returns default value of True.
        """
        do_validation = True
        if self.config.has_option('Count', 'DoValidation'):
            do_validation = self.config.getboolean('Count', 'DoValidation')
        return do_validation

    def count_num_res_steps(self):
        """Returns number of timesteps after cropping.
        """
        return self.count_num_steps() - self.count_num_steps_crop() * 2

    def train_vids(self):
        """Returns list of paths to videos in training data.
        """
        patt = os.path.join(self.train_dir(), 'videos', '*.mp4')
        return glob.glob(patt)

    def test_vids(self):
        """Returns list of paths to videos in test data.
        """
        patt = os.path.join(self.test_dir(), 'videos', '*.mp4')
        return glob.glob(patt)

    def test_truth_files(self):
        """Returns list of paths to truth files in test data.
        """
        patt = os.path.join(self.test_dir(), 'truth', '*.csv')
        return glob.glob(patt)

    def all_video_ids(self):
        """Gets all video IDs as a list.
        """
        video_ids = []
        for vid in self.train_vids():
            _, f = os.path.split(vid)
            vid_id, _ = os.path.splitext(f)
            if vid_id not in video_ids:
                video_ids.append(vid_id)
        return video_ids

    def length_path(self):
        """Returns path to length annotations.
        """
        if self.config.has_option('Data', 'LengthFormat') and \
           self.config.get('Data', 'LengthFormat') == "box":
            return os.path.join(self.train_dir(), 'boxLength.csv')
        else:
            return os.path.join(self.train_dir(), 'length.csv')

    def cover_path(self):
        """Returns path to cover annotations.
        """
        return os.path.join(self.train_dir(), 'cover.csv')

    def train_mask_imgs_dir(self):
        """Returns path to images used for training find ruler alg.
        """
        return os.path.join(self.train_dir(), 'masks', 'images')

    def train_mask_masks_dir(self):
        """Returns path to masks used for training find ruler alg.
        """
        return os.path.join(self.train_dir(), 'masks', 'masks')

    def train_imgs_dir(self):
        """Returns path to training images directory.
        """
        return os.path.join(self.work_dir(), 'train_imgs')

    def train_rois_dir(self):
        """Returns path to training roi images directory.
        """
        return os.path.join(self.work_dir(), 'train_rois')

    def train_dets_dir(self):
        """Returns path to training detection images directory.
        """
        return os.path.join(self.work_dir(), 'train_dets')

    def train_imgs(self):
        """Returns list of all training images.
        """
        patt = os.path.join(self.train_imgs_dir(), '**', '*.jpg')
        return glob.glob(patt, recursive=True)

    def num_frames_path(self):
        """Returns path to csv containing number of frames per video.
        """
        return os.path.join(self.train_imgs_dir(), 'num_frames.csv')

    def train_roi_img(self, video_id, frame):
        """Returns a specific image.
        """
        return os.path.join(
            self.train_rois_dir(),
            video_id,
            "{:04d}.jpg".format(frame)
        )

    def train_rois(self):
        """Returns list of all training roi images.
        """
        patt = os.path.join(self.train_rois_dir(), '**', '*.jpg')
        patt2 = os.path.join(self.train_rois_dir(), '**', '*.png')
        return glob.glob(patt, recursive=True).extend(glob.glob(patt2, recursive=True)

    def train_dets(self):
        """Returns list of all training detection images.
        """
        patt = os.path.join(self.train_dets_dir(), '**', '*.jpg')
        return glob.glob(patt, recursive=True)

    def checkpoints_dir(self, model):
        """Returns path to checkpoints directory.

        # Arguments
            model: Which model this corresponds to, one of find_ruler,
            detect, classify, count.
        """
        return os.path.join(self.work_dir(), 'checkpoints', model)

    def checkpoint_best(self, model):
        """Returns path to best checkpoint file.

        The path is meant to be formatted with epoch and validation loss.

        # Arguments
            model: Which model this corresponds to, one of find_ruler,
            detect, classify, count.
        """
        fname = "checkpoint-best-{epoch:03d}-{loss:.4f}.hdf5"
        return os.path.join(self.checkpoints_dir(model), fname)

    def checkpoint_periodic(self, model):
        """Returns path to periodic checkpoint file.

        The path is meant to be formatted with epoch and validation loss.

        # Arguments
            model: Which model this corresponds to, one of find_ruler,
            detect, classify, count.
        """
        fname = "checkpoint-periodic-{epoch:03d}-{loss:.4f}.hdf5"
        return os.path.join(self.checkpoints_dir(model), fname)

    def tensorboard_dir(self, model):
        """Returns path to tensorboard directory.
        """
        name = datetime.now().strftime("%Y_%m_%d_")
        name += model
        return os.path.join(self.work_dir(), 'tensorboard', name)

    def inference_dir(self):
        """Returns output path for inference results.
        """
        return os.path.join(self.work_dir(), 'inference')

    def find_ruler_inference_path(self):
        """Returns path to find ruler inference results.
        """
        return os.path.join(self.inference_dir(), 'find_ruler.csv')

    def detect_inference_path(self):
        """Returns path to detection inference results.
        """
        return os.path.join(self.inference_dir(), 'detect.csv')

    def classify_inference_path(self):
        """Returns path to classification inference results.
        """
        return os.path.join(self.inference_dir(), 'classify.csv')

    def test_output_dir(self):
        """Returns path to test outputs.
        """
        return os.path.join(self.work_dir(), 'test')
