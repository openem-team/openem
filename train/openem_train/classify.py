"""Functions for training classification algorithm.
"""

import os
import glob
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from openem_train.inception.inception import inception_model
from openem_train.inception.inception_dataset import InceptionDataset
from openem_train.util.model_utils import keras_to_tensorflow

def _save_model(config, model):
    """Loads best weights and converts to protobuf file.

    # Arguments
        config: ConfigInterface object.
        model: Keras Model object.
    """
    best = glob.glob(os.path.join(config.checkpoints_dir('classify'), '*best*'))
    latest = max(best, key=os.path.getctime)
    model.load_weights(latest)
    os.makedirs(config.classify_model_dir(), exist_ok=True)
    keras_to_tensorflow(model, ['cat_species_1', 'cat_cover_1'], config.classify_model_path())

def train(config):
    """Trains classification model.

    # Arguments
        config: ConfigInterface object.
    """
    # Create tensorboard and checkpoints directories.
    os.makedirs(config.checkpoints_dir('classify'), exist_ok=True)
    os.makedirs(config.tensorboard_dir(), exist_ok=True)

    # Build the inception model.
    model = inception_model(
        input_shape=(config.classify_height(), config.classify_width(), 3),
        num_classes=config.num_classes())

    # Set up dataset interface.
    dataset = InceptionDataset(config)

    # Define learning rate schedule.
    def schedule(epoch):
        if epoch < 1:
            return 5e-4
        if epoch < 5:
            return 3e-4
        if epoch < 10:
            return 1e-4
        if epoch < 20:
            return 5e-5
        return 1e-5

    # Set trainable layers.
    for layer in model.layers:
        layer.trainable = True

    # Set up callbacks.
    checkpoint_best = ModelCheckpoint(
        config.checkpoint_best('classify'),
        verbose=1,
        save_weights_only=False,
        save_best_only=True)

    checkpoint_periodic = ModelCheckpoint(
        config.checkpoint_periodic('classify'),
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
        dataset.generate(batch_size=config.classify_batch_size()),
        steps_per_epoch=dataset.train_batches(config.classify_batch_size()),
        epochs=config.classify_num_epochs(),
        verbose=1,
        callbacks=[
            checkpoint_best,
            checkpoint_periodic,
            tensorboard,
            lr_sched
        ],
        validation_data=dataset.generate_test(
            batch_size=config.classify_val_batch_size()
        ),
        validation_steps=dataset.test_batches(
            config.classify_val_batch_size()
        )
    )

    # Save the model.
    _save_model(config, model)
