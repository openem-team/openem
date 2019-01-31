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
