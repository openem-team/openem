"""Functions for training detection algorithm.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.applications.inception_v3 import preprocess_input
from openem_train.ssd import ssd
from openem_train.ssd.ssd_training import MultiboxLoss
from openem_train.ssd.ssd_utils import BBoxUtility
from openem_train.ssd.ssd_dataset import SSDDataset
from openem_train.util.model_utils import keras_to_tensorflow
sys.path.append('../python')
import openem


def _save_model(config, model):
    """Loads best weights and converts to protobuf file.

    # Arguments
        config: ConfigInterface object.
        model: Keras Model object.
    """
    best = glob.glob(os.path.join(config.checkpoints_dir('detect'), '*best*'))
    latest = max(best, key=os.path.getctime)
    model.load_weights(latest)
    os.makedirs(config.detect_model_dir(), exist_ok=True)
    keras_to_tensorflow(model, ['output_node0'], config.detect_model_path())

def train(config):
    """Trains detection model.

    # Arguments
        config: ConfigInterface object.
    """

    # Create tensorboard and checkpoints directories.
    os.makedirs(config.checkpoints_dir('detect'), exist_ok=True)
    os.makedirs(config.tensorboard_dir(), exist_ok=True)

    # Build the ssd model.
    model = ssd.ssd_model(
        input_shape=(config.detect_height(), config.detect_width(), 3),
        num_classes=config.num_classes())

    # Set trainable layers.
    for layer in model.layers:
        layer.trainable = True

    # Set up loss and optimizer.
    loss_obj = MultiboxLoss(
        config.num_classes(),
        neg_pos_ratio=2.0,
        pos_cost_multiplier=1.1)
    adam = Adam(lr=3e-5)

    # Compile the model.
    model.compile(loss=loss_obj.compute_loss, optimizer=adam)
    model.summary()

    # Get prior box layers from model.
    prior_box_names = [
        'conv4_3_norm_mbox_priorbox',
        'fc7_mbox_priorbox',
        'conv6_2_mbox_priorbox',
        'conv7_2_mbox_priorbox',
        'conv8_2_mbox_priorbox',
        'pool6_mbox_priorbox']
    priors = []
    for prior_box_name in prior_box_names:
        layer = model.get_layer(prior_box_name)
        if layer is not None:
            priors.append(layer.prior_boxes)
    priors = np.vstack(priors)

    # Set up bounding box utility.
    bbox_util = BBoxUtility(config.num_classes(), priors)

    # Set up dataset interface.
    dataset = SSDDataset(
        config,
        bbox_util=bbox_util,
        preproc=lambda x: x)

    # Set up keras callbacks.
    checkpoint_best = ModelCheckpoint(
        config.checkpoint_best('detect'),
        verbose=1,
        save_weights_only=False,
        save_best_only=True)

    checkpoint_periodic = ModelCheckpoint(
        config.checkpoint_periodic('detect'),
        verbose=1,
        save_weights_only=False,
        period=1)

    tensorboard = TensorBoard(
        config.tensorboard_dir(),
        histogram_freq=0,
        write_graph=True,
        write_images=True)

    # Fit the model.
    batch_size = config.detect_batch_size()
    val_batch_size = config.detect_val_batch_size()
    model.fit_generator(
        dataset.generate_ssd(
            batch_size=batch_size,
            is_training=True),
        steps_per_epoch=dataset.nb_train_samples // batch_size,
        epochs=config.detect_num_epochs(),
        verbose=1,
        callbacks=[checkpoint_best, checkpoint_periodic, tensorboard],
        validation_data=dataset.generate_ssd(
            batch_size=val_batch_size,
            is_training=False),
        validation_steps=dataset.nb_test_samples // val_batch_size,
        initial_epoch=0)

    # Save the model.
    _save_model(config, model)

def infer(config):
    """Runs detection model on extracted ROIs.

    # Arguments
        config: ConfigInterface object.
    """
    # Make a dict to contain detection results.
    det_data = {
        'video_id' : [],
        'frame' : [],
        'x' : [],
        'y' : [],
        'w' : [],
        'h' : []
    }

    # Initialize detector from deployment library.
    detector = openem.Detector()
    status = detector.Init(config.detect_model_path())
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize detector!")

    for img_path in config.train_rois():

        # Load in image.
        img = openem.Image()
        status = img.FromFile(img_path)
        if not status == openem.kSuccess:
            raise IOError("Failed to load image {}".format(p))

        # Add image to processing queue.
        status = detector.AddImage(img)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to add image for processing!")

        # Process the loaded image.
        detections = openem.VectorVectorDetection()
        status = detector.Process(detections)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to process image {}!".format(img_path))

        # Write detection to dict.
        for dets in detections:
            for det in dets:
                path, f = os.path.split(img_path)
                frame, _ = os.path.splitext(f)
                video_id = os.path.basename(os.path.normpath(path))
                x, y, w, h = det.location
                det_data['video_id'].append(video_id)
                det_data['frame'].append(frame)
                det_data['x'].append(x)
                det_data['y'].append(y)
                det_data['w'].append(w)
                det_data['h'].append(h)
        print("Finished detection on {}".format(img_path))

    # Write detections to csv.
    os.makedirs(config.inference_dir(), exist_ok=True)
    d = pd.DataFrame(det_data)
    d.to_csv(config.detect_inference_path(), index=False)
