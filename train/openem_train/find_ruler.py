"""Functions for training find ruler algorithm.
"""

import os
import glob
import sys
import pandas as pd
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from openem_train.unet.unet import unet_model
from openem_train.unet.unet_dataset import UnetDataset
from openem_train.util.model_utils import keras_to_tensorflow
sys.path.append('../python')
import openem

def _save_model(config, model):
    """Loads best weights and converts to protobuf file.

    # Arguments
        config: ConfigInterface object.
        model: Keras Model object.
    """
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
    # Create tensorboard and checkpoints directories.
    os.makedirs(config.checkpoints_dir('find_ruler'), exist_ok=True)
    os.makedirs(config.tensorboard_dir(), exist_ok=True)

    # Build the unet model.
    model = unet_model(
        input_shape=(config.find_ruler_height(), config.find_ruler_width(), 3)
    )

    # Set up dataset interface.
    dataset = UnetDataset(config)

    # Define learning rate schedule.
    def schedule(epoch):
        if epoch < 10:
            return 1e-3
        if epoch < 25:
            return 2e-4
        if epoch < 60:
            return 1e-4
        if epoch < 80:
            return 5e-5
        return 2e-5

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
        config.tensorboard_dir(),
        histogram_freq=0,
        write_graph=False,
        write_images=True)

    lr_sched = LearningRateScheduler(schedule=schedule)

    # Fit the model.
    model.summary()
    model.fit_generator(
        dataset.generate(batch_size=config.find_ruler_batch_size()),
        steps_per_epoch=60,
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
        validation_steps=len(dataset.test_idx)//config.find_ruler_val_batch_size()
    )

    # Save the model.
    _save_model(config, model)

def predict(config):
    """Runs find ruler model on video frames.

    # Arguments
        config: ConfigInterface object.
    """
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

    # Create and initialize the mask finder.
    mask_finder = openem.RulerMaskFinder()
    status = mask_finder.Init(config.find_ruler_model_path())
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize ruler mask finder!")

    for img_path in config.train_imgs():

        print("Finding mask for image {}...".format(img_path))

        # Get video id from path.
        path, fname = os.path.split(img_path)
        frame, _ = os.path.splitext(fname)
        video_id = os.path.basename(os.path.normpath(path))

        # Load in image.
        img = openem.Image()
        status = img.FromFile(img_path)
        if not status == openem.kSuccess:
            raise IOError("Failed to load image {}".format(img_path))

        # Add image to processing queue.
        status = mask_finder.AddImage(img)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to add image {} for processing!".format(img_path))

        # Process the loaded image.
        masks = openem.VectorImage()
        status = mask_finder.Process(masks)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to process image {}!".format(img_path))

        # Resize the mask back to the same size as the image.
        mask = masks[0]
        mask.Resize(img.Width(), img.Height())

        # Initialize the mean mask if necessary.
        if video_id not in mask_avg:
            mask_avg[video_id] = np.zeros((img.Height(), img.Width()));

        # Add the mask to the mask average.
        mask_data = mask.DataCopy()
        mask_data = np.array(mask_data)
        mask_data = np.reshape(mask_data, (img.Height(), img.Width()))
        mask_avg[video_id] += mask_data

    for vid_id in mask_avg:

        print("Finding ruler endpoints for video {}...".format(vid_id))

        # Convert mask image from numpy to openem format.
        mask_vec = mask_avg[video_id].copy()
        mask_vec = mask_vec / np.max(mask_vec)
        mask_vec = mask_vec * 255.0
        mask_vec = mask_vec.reshape(-1).astype(np.uint8).tolist()
        mask_img = openem.Image()
        mask_img.FromData(mask_vec, img.Width(), img.Height(), 1);

        # Get ruler endpoints from the mask averages.
        p1, p2 = openem.RulerEndpoints(mask_img)
        find_ruler_data['video_id'].append(vid_id)
        find_ruler_data['x1'].append(p1[0])
        find_ruler_data['y1'].append(p1[1])
        find_ruler_data['x2'].append(p2[0])
        find_ruler_data['y2'].append(p2[1])

    # Write detections to csv.
    os.makedirs(config.inference_dir(), exist_ok=True)
    d = pd.DataFrame(find_ruler_data)
    d.to_csv(config.find_ruler_inference_path(), index=False)
