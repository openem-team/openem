""" Define base classes for openem models """
import logging
import os

import tensorflow as tf
import numpy as np
import cv2
from .optimizer import optimizeGraph
from multiprocessing import Queue, RawArray, Value
import ctypes

logger = logging.getLogger(__name__)

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
    input_tensor = None
    input_shape = None
    output_tensor = None
    gpu_pid = None
    batch_size = None

    def __init__(self, model_path,
                 image_dims = None,
                 gpu_fraction = 1.0,
                 input_name = 'input_1:0',
                 output_name = 'output_node0:0',
                 optimize = True,
                 optimizer_args = None,
                 batch_size = 1):
        """ Initialize an image model object
        model_path : str or path-like object
                     Path to the frozen protobuf of the tensorflow graph
        image_dims : tuple
                     Tuple for image dims: (<height>, <width>, <channels>)
                     If None, is inferred from the graph.
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
        batch_size : int
                     Maximum number of images to process as a batch
        """

        self.gpu_pid = os.getpid()
        self.batch_size = batch_size

        # Create session first with requested gpu_fraction parameter
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self.tf_session = tf.compat.v1.Session(config=config)

        with tf.io.gfile.GFile(model_path, 'rb') as graph_file:
            # Load graph off of disk into a graph definition
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(graph_file.read())

            if optimize:
                if type(output_name) == list:
                    sensitive_nodes = output_name
                else:
                    sensitive_nodes = [output_name]
                graph_def = optimizeGraph(graph_def,
                                          sensitive_nodes,
                                          optimizer_args)
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

            if image_dims is None:
                image_dims = (self.input_shape[1],
                              self.input_shape[2],
                              self.input_shape[3])
                print(f"Inferred image dims = {image_dims}")
            # Each channel is 1 byte, width * height * # of channels
            image_size_in_bytes = image_dims[0] * image_dims[1] * image_dims[2]
            # Initialize the shared memory buffer
            buffer_count=batch_size * 4
            self._buffers=[]
            self._inputQueue = Queue(maxsize=buffer_count)
            self._processQueue = Queue(maxsize=buffer_count)
            for buffer_num in range(buffer_count):
                self._buffers.append(RawArray(ctypes.c_uint8, image_size_in_bytes*8))
                self._inputQueue.put(buffer_num)


    def inputShape(self):
        """ Returns the shape of the input image for this network """
        return self.input_shape

    def _addImage(self, image, preprocessor, cookie=None):
        """ Adds an image into the next to process batch
            image: np.ndarray
                   Image data to add into the batch
            preprocessor: models.Preprocessor
                   Preprocessing logic to apply to image prior to insertion
            cookie: Extra info to pass back to caller based on image
        """
        processed_image = preprocessor(image,
                                       self.inputShape()[2],
                                       self.inputShape()[1])
        
        idx = self._inputQueue.get()
        flat = np.frombuffer(self._buffers[idx])
        flat[:] = processed_image.reshape(-1)
        self._processQueue.put((idx, cookie))

    def process(self, batch_size=None):
        """ Process the current batch of image(s).

        Returns None if there are no images.
        """
        if os.getpid() != self.gpu_pid:
            logger.error("Tensorflow crossed process boundary")
            return None,None

        if batch_size is None:
            # Default to whatever is ready to process
            batch_size = self._processQueue.qsize()

        if batch_size == 0:
            return None,None

        images=[]
        image_indices=[]
        image_cookies=[]
        for idx in range(batch_size):
            msg = self._processQueue.get()
            if msg is not None:
                image_idx = msg[0]
                cookie = msg[1]
                image_indices.append(image_idx)
                image_cookies.append(cookie)
                flat = np.frombuffer(self._buffers[image_idx])
                images.append(flat.reshape(self.inputShape()[1:]))
        result = self.tf_session.run(
            self.output_tensor,
            feed_dict={self.input_tensor: np.array(images)})

        # Return image buffers to the free queue
        for idx in image_indices:
            self._inputQueue.put(idx)
        return result, image_cookies
