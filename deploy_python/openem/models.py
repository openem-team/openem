""" Define base classes for openem models """
import tensorflow as tf
import numpy as np
import cv2
from .optomizer import optomizeGraph

class Preprocessor:
    def __init__(self, scale=None, bias=None, rgb=None):
        """ Create a preprocessing object to handle image transformations
            required for network to run

        scale : float
                Scale factor applied to image after conversion to float
        bias : 3-element np.array
               Bias applied to image after scaling
        rgb : bool
              Set to true to convert 3 channel data to RGB (from BGR)
        """
        self.scale = scale
        self.bias = bias
        self.rgb = rgb

    def __call__(self, image, requiredWidth, requiredHeight):
        """ Run the required preprocessing steps on an input image
        image : np.ndarray containing the image data
        """
        # Resize the image first
        if image.shape[0] != requiredHeight or image.shape[1] != requiredWidth:
            image = cv2.resize(image, (requiredWidth, requiredHeight))

        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert the image to a floating point value
        image = image.astype(np.float32)

        if self.scale is not None:
            image *= self.scale
        if self.bias is not None:
            image += self.bias

        return image

class ImageModel:
    """ Base class for serving image-related models from tensorflow """
    tf_session = None
    images = None
    input_tensor = None
    input_shape = None
    output_tensor = None
    def __init__(self, model_path, gpu_fraction=1.0,
                 input_name = 'input_1:0',
                 output_name = 'output_node0:0',
                 optomize = True,
                 optomizer_args = None):
        """ Initialize an image model object
        model_path : str or path-like object
                     Path to the frozen protobuf of the tensorflow graph
        gpu_fraction : float
                       Fraction of GPU allowed to be used by this object.
        input_name : str
                     Name of the tensor that serves as the image input
        output_name : str or list of str
                      Name(s) of the the tensor that serves as the output of
                      the network. If a singular tensor is given; then the
                      process function will return that singular tensor. Else
                      the process function returns each tensor output in the
                      order specified in this function as a list.
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

            if optomize:
                if type(output_name) == list:
                    sensitive_nodes = output_name
                else:
                    sensitive_nodes = [output_name]
                graph_def = optomizeGraph(graph_def,
                                          sensitive_nodes,
                                          optomizer_args)
            if type(output_name) == list:
                return_elements = [input_name, *output_name]
                tensors = tf.import_graph_def(
                    graph_def,
                    return_elements=return_elements)
                # The first is an input
                self.input_tensor = tensors[0]
                # The rest are outputs
                self.output_tensor = tensors[1:]
            else:
                return_elements = [input_name, output_name]
                self.input_tensor, self.output_tensor = tf.import_graph_def(
                    graph_def,
                    return_elements=return_elements)

            self.input_shape = self.input_tensor.get_shape().as_list()

    def inputShape(self):
        """ Returns the shape of the input image for this network """
        return self.input_shape

    def _addImage(self, image, preprocessor):
        """ Adds an image into the next to process batch
            image: np.ndarray
                   Image data to add into the batch
            preprocessor: models.Preprocessor
                   Preprocessing logic to apply to image prior to insertion
        """
        if self.images == None:
            self.images = []

        processed_image = preprocessor(image,
                                       self.inputShape()[2],
                                       self.inputShape()[1])

        self.images.append(processed_image)

    def process(self):
        """ Process the current batch of image(s).

        Returns None if there are no images.
        """
        if self.images == None:
            return None

        result = self.tf_session.run(
            self.output_tensor,
            feed_dict={self.input_tensor: np.array(self.images)})
        self.images = None
        return result
