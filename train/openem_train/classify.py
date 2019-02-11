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

"""Functions for training classification algorithm.
"""

import os
import glob
import sys
import pandas as pd

def _save_model(config, model):
    """Loads best weights and converts to protobuf file.

    # Arguments
        config: ConfigInterface object.
        model: Keras Model object.
    """
    from openem_train.util.model_utils import keras_to_tensorflow
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
    # Import keras.
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.callbacks import LearningRateScheduler
    from openem_train.inception.inception import inception_model
    from openem_train.inception.inception_dataset import InceptionDataset
    from openem_train.util.utils import find_epoch

    # Create tensorboard and checkpoints directories.
    tensorboard_dir = config.tensorboard_dir('classify')
    os.makedirs(config.checkpoints_dir('classify'), exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Build the inception model.
    model = inception_model(
        input_shape=(config.classify_height(), config.classify_width(), 3),
        num_classes=config.num_classes())

    # If initial epoch is nonzero we load the model from checkpoints 
    # directory.
    initial_epoch = config.classify_initial_epoch()
    if initial_epoch != 0:
        checkpoint = find_epoch(
            config.checkpoints_dir('classify'),
            initial_epoch
        )
        model.load_weights(checkpoint)

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
        tensorboard_dir,
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
        ),
        initial_epoch=initial_epoch
    )

    # Save the model.
    _save_model(config, model)

def predict(config):
    """Runs detection model on extracted detections.

    # Arguments
        config: ConfigInterface object.
    """
    # Import deployment library.
    sys.path.append('../python')
    import openem

    # Make a dict to contain detection results.
    class_data = {
        'video_id' : [],
        'frame' : [],
        'no_fish' : [],
        'covered' : [],
        'clear' : []
    }
    species_list = ['_',] + config.species()
    for spc in species_list:
        class_data['species_' + spc] = []

    # Initialize detector from deployment library.
    classifier = openem.Classifier()
    status = classifier.Init(config.classify_model_path())
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize detector!")
    w, h = classifier.ImageSize()

    for img_path in config.train_dets():

        # Get video id from path.
        path, fname = os.path.split(img_path)
        frame, _ = os.path.splitext(fname)
        video_id = os.path.basename(os.path.normpath(path))


        # Load in image.
        img = openem.Image()
        status = img.FromFile(img_path)
        if not status == openem.kSuccess:
            raise IOError("Failed to load image {}".format(img_path))
        img.Resize(w, h)

        # Add image to processing queue.
        status = classifier.AddImage(img)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to load image {}".format(img_path))

        # Process loaded image.
        scores = openem.VectorClassification()
        status = classifier.Process(scores)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to process image {}!".format(img_path))

        # Write classification results.
        for score in scores:
            class_data['video_id'].append(video_id)
            class_data['frame'].append(int(frame))
            for spc, spc_score in zip(species_list, score.species):
                class_data['species_' + spc].append(spc_score)
            class_data['no_fish'].append(score.cover[0])
            class_data['covered'].append(score.cover[1])
            class_data['clear'].append(score.cover[2])
        print("Finished classification on {}".format(img_path))

    # Write classification results to csv.
    os.makedirs(config.inference_dir(), exist_ok=True)
    d = pd.DataFrame(class_data)
    d.to_csv(config.classify_inference_path(), index=False)
