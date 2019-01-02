"""Functions for training find ruler algorithm.
"""

import os
import glob
import sys
import pandas as pd
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
