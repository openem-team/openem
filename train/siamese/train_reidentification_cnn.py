#!/usr/bin/env python

import argparse
import os
import numpy as np
from functools import partial
from multiprocessing import Queue
from multiprocessing import Process
from utilities import is_valid_path
from utilities import ModelData
from utilities import rate
from utilities import contrastive_loss
from utilities import noop
from utilities import get_session
from keras.layers import Dense
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import linear
from keras.initializers import Constant
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model
from keras.constraints import min_max_norm
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K

def euclidean(vecs):
    """ Returns the euclidean distance between vectors.
        Inputs:
        vecs - Tuple of two vectors.
    """
    x, y = vecs
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def euclidean_shape(shapes):
    """ Returns shape of euclidean distance output.
        Inputs:
        shapes - Tuple of two shapes.
    """
    shape0, shape1 = shapes
    return (shape0[0], 1)

def provider(data_provider):
    """ Pushes data onto the queue.
        Inputs:
        data_provider - A DataProvider object.
    """
    data_provider.start()

def generator(batch_queue):
    """ Yields data batches from a queue.
        Inputs:
        batch_queue - Queue containing tuples of image pairs and labels.
    """
    while True:
        yield batch_queue.get(True, None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train appearance feature extraction CNN " +
        "for reidentification task")
    parser.add_argument("model_dir",
        type=lambda x: is_valid_path(parser, x),
        help="Directory containing reidentification data")
    parser.add_argument("--feature_vector_length",
        type=int,
        default=500,
        help="Size of output feature vector")
    parser.add_argument("--holdout_fraction",
        type=float,
        default=0.1,
        help="Fraction of total number of pairs to use for verification")
    parser.add_argument("--dropout_fraction",
        type=float,
        default=0.1,
        help="Fraction for use in dropout layer.")
    parser.add_argument("--num_epochs",
        type=int,
        default=30,
        help="Number of training epochs")
    parser.add_argument("--batch_size",
        type=int,
        default=32,
        help="Batch size")
    parser.add_argument("--min_norm",
        type=float,
        default=0.0,
        help="Min norm constraint on last layer.")
    parser.add_argument("--max_norm",
        type=float,
        default=0.6,
        help="Max norm constraint on last layer.")
    parser.add_argument("--rate_init",
        type=float,
        default=0.001,
        help="Initial learning rate.")
    parser.add_argument("--steps_per_epoch",
        type=int,
        default=5000,
        help="Number of steps per epoch, determined automatically if not set.")
    parser.add_argument(
        '--val_steps_per_epoch',
        help="Number of validation steps per epoch.",
        type=int,
        default=500
    )
    parser.add_argument("--epochs_per_order",
        type=float,
        default=10.0,
        help="Number of epochs to drop rate by an order of magnitude.")
    parser.add_argument("--do_plots",
        action="store_true",
        help="Do spot check plots for each batch")
    parser.add_argument("--weights_init",
        type=lambda x: is_valid_path(parser, x),
        default=None,
        help="Path to weights file used to initialize model.")
    parser.add_argument(
        '--tensorboard_dir',
        help="Directory to use for tensorboard outputs.",
        type=str,
    )

    # Parameters for LR scheduler
    parser.add_argument('--lr-monitor', default='loss', help='Quantity to be monitored')
    parser.add_argument('--lr-factor', default=0.1, type=float, help='Factor by which the learning rate will be reduced. new_lr = lr * factor')
    parser.add_argument('--lr-patience', default=3, type=int, help="number of epochs that produced the monitored quantity with no improvement after which training will be stopped. Validation quantities may not be produced for every epoch, if the validation frequency (model.fit(validation_freq=5)) is greater than one.")
    parser.add_argument('--lr-verbose', default=1, choices=[1,2], help='update messages')
    parser.add_argument('--lr-mode', default='min', choices=['min','max','auto'],
                         help="In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing; in auto mode, the direction is automatically inferred from the name of the monitored quantity.")
    parser.add_argument('--lr-min-delta', default=0.01, help="threshold for measuring the new optimum, to only focus on significant changes.", type=float)
    parser.add_argument('--lr-cooldown', default=0, help="number of epochs to wait before resuming normal operation after lr has been reduced.", type=float)
    parser.add_argument('--lr-min-lr', default=1e-8, help="lower bound on the learning rate.", type=float)

    args = parser.parse_args()
    get_session()

    # Set up model data interface.
    model_data = ModelData(args.model_dir)

    # Create checkpoint and tensorboard directories.
    os.makedirs(model_data.cnn_checkpoints_dir(), exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)

    # Set up separate processes to push image pairs onto queue
    train_queue = Queue(10)
    validate_queue = Queue(10)
    augmentations = [noop, np.fliplr, np.flipud, np.rot90]
    train_provider, validate_provider = model_data.data_providers(
        augmentations,
        args.batch_size,
        train_queue,
        validate_queue,
        args.do_plots)
    train_process = Process(target=provider, args=(train_provider,))
    validate_process = Process(target=provider, args=(validate_provider,))
    train_process.daemon = True
    validate_process.daemon = True
    train_process.start()
    validate_process.start()

    img_shape = (224, 224, 3)

    # Create model
    model = ResNet50(include_top=False, input_shape=img_shape)

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    x = Dense(
        args.feature_vector_length,
        activation="linear",
        kernel_initializer="he_normal")(x)
    x = LeakyReLU(alpha=0.1)(x)
    model = Model(inputs=model.input, outputs=x)

    # Set all layers to trainable
    for layer in model.layers:
        layer.trainable = True
    model.summary()

    # Set up siamese network
    input0 = Input(shape=img_shape)
    input1 = Input(shape=img_shape)
    output0 = model(input0)
    output1 = model(input1)
    distance = Lambda(
        euclidean,
        output_shape=euclidean_shape
    )([output0, output1])
    out = Dense(1,
        activation="sigmoid",
        bias_initializer=Constant(value=-10.0)
    )(distance)
    s_model = Model(inputs=[input0, input1], outputs=out)
    if args.weights_init:
        s_model.load_weights(args.weights_init)
    s_model.summary()

    # Compile
    s_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['acc', 'mae'])

    # Save the architecture
    model_data.save_architecture("cnn", s_model)

    # Set up callbacks.
    checkpoint_best = ModelCheckpoint(
        model_data.cnn_checkpoint_best(),
        verbose=1,
        save_weights_only=False,
        save_best_only=True)

    checkpoint_periodic = ModelCheckpoint(
        model_data.cnn_checkpoint_periodic(),
        verbose=1,
        save_weights_only=False,
        period=1)

    tensorboard = TensorBoard(
        args.tensorboard_dir,
        histogram_freq=0,
        write_graph=False,
        write_images=True)

    scheduler = partial(
        rate,
        rate_init=args.rate_init,
        epochs_per_order=args.epochs_per_order
    )

    lr_scheduler = ReduceLROnPlateau(
        monitor=args.lr_monitor,
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=args.lr_verbose,
        mode=args.lr_mode,
        min_delta=args.lr_min_delta,
        cooldown=args.lr_cooldown,
        min_lr=args.lr_min_lr)

    # Fit the model.
    s_model.fit_generator(
        generator(train_queue),
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.num_epochs,
        verbose=1,
        callbacks=[
            checkpoint_best,
            checkpoint_periodic,
            tensorboard,
            lr_scheduler
        ],
        validation_data=generator(validate_queue),
        validation_steps=args.val_steps_per_epoch
    )
