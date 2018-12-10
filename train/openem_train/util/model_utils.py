"""Utilities for working with models."""

import os
from keras import backend as K
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def keras_to_tensorflow(model, pred_node_names, out_path):
    """Converts a keras model to a tensorflow graph.

    # Arguments
        model: Keras model object.
        pred_node_names: Names of prediction nodes in the model.
        out_path: Path to output file in *.pb (protobuf) format.
    """
    K.set_learning_phase(0)
    K.set_image_data_format('channels_last')
    # pylint: disable=unused-variable
    pred = [tf.identity(model.outputs[i], name=p)
            for i, p in enumerate(pred_node_names)]
    sess = K.get_session()
    out_dir, out_file = os.path.split(out_path)
    constant_graph = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(add_shapes=True), pred_node_names)
    graph_io.write_graph(constant_graph, out_dir, out_file, as_text=False)
