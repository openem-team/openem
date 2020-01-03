#!/usr/bin/env python3

"""
Converts a keras model (*.h5) into a protobuf suitable to launch with
    the `infer.py` or `openem.Detect.Retinanet` directly.
"""
import argparse
import os
import keras
from keras import backend as K
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.keras_version import check_keras_version
from keras.layers import Input
from keras.models import Model
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

import tensorflow.contrib.tensorrt as trt

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


def get_session(check):
    if not check:
        config = tf.ConfigProto(device_count = {'GPU' : 0})
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    return tf.Session(config=config)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input file (*.h5)")
    parser.add_argument("output", help="Output file (*.pb)")
    parser.add_argument("--resnet", action="store_true", help="Use this for resnet-based models like RetinaNet")
    parser.add_argument("--appearance", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check the graph can load and be optomized by tensorrt (requires GPU session!)")
    args = parser.parse_args()
    check_keras_version()
    keras.backend.tensorflow_backend.set_session(get_session(args.check))
    K.set_learning_phase(0)
    if args.resnet:
        model = keras.models.load_model(args.input, custom_objects=custom_objects)
    else:
        model = keras.models.load_model(args.input)

    model.summary()
    if args.appearance:
        trk_fea = Input(shape=(500,))
        det_fea = Input(shape=(500,))
        out = model.get_layer("lambda_1")([trk_fea, det_fea])
        out = model.get_layer("dense_4")(out)
        model = Model(inputs=[trk_fea, det_fea], outputs=out)

    outputnames=[]
    for output in model.outputs:
        outputnames.append(output.name.split(':')[0])
    print(outputnames)
    keras_to_tensorflow(model, outputnames, args.output)

    if args.check:
        graph_def=tf.GraphDef()
        tensornames=[]
        for name in outputnames:
            tensornames.append(f"{name}:0")

        with open(args.output, 'rb') as graph_file:
            graph_def.ParseFromString(graph_file.read())
        trt_model=trt.create_inference_graph(graph_def,
                                             tensornames,
                                             is_dynamic_op=True,
                                             precision_mode='fp16')
