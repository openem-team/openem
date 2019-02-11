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

"""Functions for training counting algorithm.
"""

import os
import glob

def _save_model(config, model):
    """Loads best weights and converts to protobuf file.

    # Arguments
        config: ConfigInterface object.
        model: Keras Model object.
    """
    from openem_train.util.model_utils import keras_to_tensorflow
    best = glob.glob(os.path.join(config.checkpoints_dir('count'), '*best*'))
    latest = max(best, key=os.path.getctime)
    model.load_weights(latest)
    os.makedirs(config.count_model_dir(), exist_ok=True)
    keras_to_tensorflow(model, ['cumsum_values_1'], config.count_model_path())

def train(config):
    """Trains counting model.

    # Arguments
        config: ConfigInterface object.
    """
    # Import keras.
    from keras.callbacks import ModelCheckpoint
    from keras.callbacks import TensorBoard
    from keras.callbacks import LearningRateScheduler
    from openem_train.rnn.rnn_dataset import RNNDataset
    from openem_train.rnn.rnn import rnn_model
    from openem_train.util.utils import find_epoch

    # Create the dataset.
    dataset = RNNDataset(config)

    # Build the model.
    model = rnn_model(
        input_shape=(config.count_num_steps(), config.count_num_features()),
        num_steps_crop=config.count_num_steps_crop()
    )
    model.summary()

    # If initial epoch is nonzero we load the model from checkpoints 
    # directory.
    initial_epoch = config.count_initial_epoch()
    if initial_epoch != 0:
        checkpoint = find_epoch(
            config.checkpoints_dir('count'),
            initial_epoch
        )
        model.load_weights(checkpoint)

    # Create checkpoint and tensorboard directories.
    tensorboard_dir = config.tensorboard_dir('count')
    os.makedirs(config.checkpoints_dir('count'), exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Define learning rate schedule.
    def schedule(epoch):
        if epoch < 10:
            return 5e-4
        if epoch < 20:
            return 2e-4
        if epoch < 50:
            return 1e-4
        if epoch < 100:
            return 5e-5
        return 2e-5

    # Set trainable layers.
    for layer in model.layers:
        layer.trainable = True

    # Set up callbacks.
    checkpoint_best = ModelCheckpoint(
        config.checkpoint_best('count'),
        verbose=1,
        save_weights_only=False,
        save_best_only=True)

    checkpoint_periodic = ModelCheckpoint(
        config.checkpoint_periodic('count'),
        verbose=1,
        save_weights_only=False,
        period=1)

    tensorboard = TensorBoard(
        tensorboard_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=True)

    lr_sched = LearningRateScheduler(schedule=schedule)

    # Fit the model.
    model.summary()
    model.fit_generator(
        dataset.generate(batch_size=config.count_batch_size()),
        steps_per_epoch=512,
        epochs=config.count_num_epochs(),
        verbose=1,
        callbacks=[
            checkpoint_best,
            checkpoint_periodic,
            tensorboard,
            lr_sched
        ],
        validation_data=dataset.generate_test(
            batch_size=config.count_val_batch_size()
        ),
        validation_steps=dataset.test_batches(
            config.count_val_batch_size()
        ),
        initial_epoch=initial_epoch
    )

    # Save the model.
    _save_model(config, model)
