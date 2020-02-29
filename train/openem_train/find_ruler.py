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

"""Functions for training find ruler algorithm.
"""

import os
import glob
import sys
from collections import defaultdict
import pandas as pd
import numpy as np
import cv2

def _save_model(config, model):
    """Loads best weights and converts to protobuf file.

    # Arguments
        config: ConfigInterface object.
        model: Keras Model object.
    """
    from openem_train.util.model_utils import keras_to_tensorflow
    best = glob.glob(os.path.join(config.checkpoints_dir('find_ruler'), '*best*'))
    latest = max(best, key=os.path.getctime)
    model.load_weights(latest)
    os.makedirs(config.find_ruler_model_dir(), exist_ok=True)
    keras_to_tensorflow(model, ['output_node0'], config.find_ruler_model_path())

def train(config):
    """Trains find ruler model.

    # Arguments
        config: ConfigInterface object.
    """
    # Import keras.
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import TensorBoard
    from keras.callbacks import ReduceLROnPlateau
    from openem_train.unet.unet import unet_model
    from openem_train.unet.unet_dataset import UnetDataset
    from openem_train.util.utils import find_epoch

    # Create tensorboard and checkpoints directories.
    tensorboard_dir = config.tensorboard_dir('find_ruler')
    os.makedirs(config.checkpoints_dir('find_ruler'), exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Build the unet model.
    num_channels = config.find_ruler_num_channels()
    model = unet_model(
        input_shape=(
            config.find_ruler_height(),
            config.find_ruler_width(),
            num_channels,
        )
    )

    # If initial epoch is nonzero we load the model from checkpoints 
    # directory.
    initial_epoch = config.find_ruler_initial_epoch()
    if initial_epoch != 0:
        checkpoint = find_epoch(
            config.checkpoints_dir('find_ruler'),
            initial_epoch
        )
        model.load_weights(checkpoint)

    # Set up dataset interface.
    dataset = UnetDataset(config)

    # Set trainable layers.
    for layer in model.layers:
        layer.trainable = True

    # Set up callbacks.
    checkpoint_best = ModelCheckpoint(
        config.checkpoint_best('find_ruler'),
        verbose=1,
        save_weights_only=False,
        save_best_only=True)

    checkpoint_periodic = ModelCheckpoint(
        config.checkpoint_periodic('find_ruler'),
        verbose=1,
        save_weights_only=False,
        period=1)

    tensorboard = TensorBoard(
        tensorboard_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=True)

    lr_sched = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10,
        min_lr=0,
    )

    # Fit the model.
    model.summary()
    num_steps = int(config.find_ruler_steps_per_epoch() / 10)
    num_steps = min(num_steps, len(dataset.test_idx) // config.find_ruler_val_batch_size())
    model.fit_generator(
        dataset.generate(batch_size=config.find_ruler_batch_size()),
        steps_per_epoch=config.find_ruler_steps_per_epoch(),
        epochs=config.find_ruler_num_epochs(),
        verbose=1,
        callbacks=[
            checkpoint_best,
            checkpoint_periodic,
            tensorboard,
            lr_sched
        ],
        validation_data=dataset.generate_validation(
            batch_size=config.find_ruler_val_batch_size()
        ),
        validation_steps=num_steps,
        initial_epoch=initial_epoch
    )

    # Save the model.
    _save_model(config, model)

def predict(config):
    """Runs find ruler model on video frames.

    # Arguments
        config: ConfigInterface object.
    """
    # Import deployment library.
    sys.path.append('../python')
    import openem
    from openem.FindRuler import RulerMaskFinder

    # Make a dict to contain find ruler results.
    find_ruler_data = {
        'video_id' : [],
        'x1' : [],
        'y1' : [],
        'x2' : [],
        'y2' : [],
    }

    # Make a dict to store mean of all masks.
    mask_avg = {}

    # Make a dict to store number of masks found per video.
    num_masks = defaultdict(int)

    # Get number of channels.
    num_channels = config.find_ruler_num_channels()

    # Create and initialize the mask finder.
    image_dims = (config.find_ruler_width(), config.find_ruler_height(), num_channels)
    finder = RulerMaskFinder(config.find_ruler_model_path(), image_dims)

    # Check if saving masks is enabled.
    save_masks = config.find_ruler_save_masks()
    if save_masks:
        mask_dir = config.predict_masks_dir()
        os.makedirs(mask_dir, exist_ok=True)

    for img_path in config.train_imgs():

        # Get video id from path.
        path, fname = os.path.split(img_path)
        frame, _ = os.path.splitext(fname)
        video_id = os.path.basename(os.path.normpath(path))

        if num_masks[video_id] > 200:
            continue
        else:
            num_masks[video_id] += 1

        print("Finding mask for image {}...".format(img_path))

        # Load in image.
        if num_channels == 3:
            img = cv2.imread(img_path)
        elif num_channels == 4:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # Add image to processing queue.
        finder.addImage(img)

        # Process the image.
        image_result = finder.process()
        if image_result is None:
            raise RuntimeError(f"Failed to process image {img_path}!")

        # Resize the mask back to the same size as the image.
        mask = image_result[0]
        h, w, _ = img.shape
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)

        # If specified, save the generated mask.
        if save_masks:
            basename = os.path.basename(img_path)
            subdir = os.path.basename(os.path.dirname(img_path))
            os.makedirs(os.path.join(mask_dir, subdir), exist_ok=True)
            mask_path = os.path.join(mask_dir, subdir, basename)
            cv2.imwrite(mask_path, mask)

        # Initialize the mean mask if necessary.
        if video_id not in mask_avg:
            mask_avg[video_id] = np.zeros((h, w));

        # Add the mask to the mask average.
        mask_data = np.copy(mask)
        mask_data = np.array(mask_data)
        mask_data = np.reshape(mask_data, (h, w))
        mask_avg[video_id] += mask_data

    for video_id in mask_avg:

        print("Finding ruler endpoints for video {}...".format(video_id))

        # Convert mask image from numpy to openem format.
        mask_vec = mask_avg[video_id].copy()
        mask_vec = mask_vec / np.max(mask_vec)
        mask_vec = mask_vec * 255.0
        mask_vec = mask_vec.reshape(-1).astype(np.uint8).tolist()
        mask_img = openem.Image()
        mask_img.FromData(mask_vec, img.Width(), img.Height(), 1);

        # Get ruler endpoints from the mask averages.
        p1, p2 = openem.RulerEndpoints(mask_img)
        x1, y1 = p1
        x2, y2 = p2
        find_ruler_data['video_id'].append(video_id)
        find_ruler_data['x1'].append(x1)
        find_ruler_data['y1'].append(y1)
        find_ruler_data['x2'].append(x2)
        find_ruler_data['y2'].append(y2)

    # Write detections to csv.
    os.makedirs(config.inference_dir(), exist_ok=True)
    d = pd.DataFrame(find_ruler_data)
    d.to_csv(config.find_ruler_inference_path(), index=False)
