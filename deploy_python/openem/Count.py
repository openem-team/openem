""" Module for finding keyframes """
import tensorflow as tf
import numpy as np
import cv2

from openem.models import ImageModel
from openem.models import Preprocessor
from openem.image import crop

class KeyframeFinder:
    def __init__(self, model_path, img_width, img_height, gpu_fraction=1.0):
        """ Initialize an image model object
        model_path : str or path-like object
                     Path to the frozen protobuf of the tensorflow graph
        img_width: Width of the image input to detector (pixels)
        img_height: Height of image input to decttor (pixels)
        gpu_fraction : float
                       Fraction of GPU allowed to be used by this object.
        """
         # Create session first with requested gpu_fraction parameter
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.tf_session = tf.compat.v1.Session(config=config)

        with tf.io.gfile.GFile(model_path, 'rb') as graph_file:
            # Load graph off of disk into a graph definition
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(graph_file.read())
            self.input_tensor, self.output_tensor = tf.import_graph_def(
                    graph_def,
                    return_elements=['input_1:0', 'cumsum_values_1:0'])

    def process(self):
        pass
