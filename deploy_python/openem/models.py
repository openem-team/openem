""" Define base classes for openem models """
import tensorflow as tf
import numpy as np

class Preprocessor:
    def __init__(scale=None, bias=None, rgb=None):
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

    def run(image, requiredWidth, requiredHeight):
        """ Run the required preprocessing steps on an input image 
        image : np.ndarray containing the image data
        """
        # Extract only 3 channels of data
        image = image.astype(np.float32)
        
        
        
class ImageModel:
    """ Base class for serving image-related models from tensorflow """
    tf_session = None
    images = None
    def __init__(self, model_path, gpu_fraction=1.0,
                 input_name = 'input_1',
                 output_name = 'output_node0:0'):
        """ Initialize an image model object 
        model_path : str or path-like object
                     Path to the frozen protobuf of the tensorflow graph
        gpu_fraction : float
                       Fraction of GPU allowed to be used by this object.
        input_name : str 
                     Name of the tensor that serves as the image input
        output_name : str
                      Name of the the tensor that serves as the output of
                      the network
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

            # TODO: place trt optomization, if elected, here

            self.input_tensor, self.output_tensor = tf.import_graph_def(
                graph_def,
                return_elements=[input_name, output_name]
            )

    def addImage(self, image, preprocessor):
        if self.images == None:
            self.images = []
        
            
