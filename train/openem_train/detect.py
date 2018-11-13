import os
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from openem_train.ssd import ssd
from openem_train.ssd.ssd_training import MultiboxLoss
from openem_train.ssd.ssd_utils import BBoxUtility
from openem_train.ssd.ssd_dataset import SSDDataset

def train(config):
    """Trains detection model.

    # Arguments
        config: ConfigParser object.
    """

    # Create tensorboard and checkpoints directories.
    os.makedirs(config.checkpoints_dir(), exist_ok=True)
    os.makedirs(config.tensorboard_dir(), exist_ok=True)

    # Build the ssd model.
    model = ssd.ssd_model((config.detect_height(), config.detect_width(), 3))

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
        preprocess_input=lambda x: x)

    # Set up keras callbacks.
    checkpoint_best = ModelCheckpoint(
        config.checkpoint_best(),
        verbose=1,
        save_weights_only=False,
        save_best_only=True)

    checkpoint_periodic = ModelCheckpoint(
        config.checkpoint_periodic(),
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

