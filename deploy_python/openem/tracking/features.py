import tensorflow as tf
import numpy as np
from openem.models import ImageModel
from openem.optimizer import optimizeGraph
import cv2

class FeaturesPreprocessor:
    """ Perform preprocessinig for RetinaNet inputs
        Meets the callable interface of openem.Detect.Preprocessor
    """
    def __init__(self,meanImage=None):
        self.mean_image = meanImage

    def __call__(self, image, requiredWidth, requiredHeight):
        #TODO: (Provide way to optionally convert channel ordering?)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        resized_image = cv2.resize(image, (requiredWidth, requiredHeight))
        if self.mean_image:
            for dim in [0,1,2]:
                resized_image[:,:,dim] -= self.mean_image[:,:,dim]
        else:
            # Use the ImageNet mean image by default; which in BGR is:
            imagenet_mean = np.array([103.939, 116.779, 123.68 ])
            resized_image -= imagenet_mean
        return resized_image / 127.5

class FeaturesExtractor(ImageModel):
    """ Feature extractor model """
    def __init__(self, modelPath, **kwargs):
        super(FeatureExtractor,self).__init__(modelPath,
                                              input_name='input_2:0',
                                              output_name='model_1/leaky_re_lu_1/LeakyRelu:0',
                                              **kwargs)
        self.preprocessor = FeaturesPreprocessor()
    def addImage(self, image, cookie=None):
        return super(FeatureExtractor, self)._addImage(image,
                                                       self.preprocessor,
                                                       cookie)

class FeaturesComparator:
    def __init__(self, model_path, gpu_fraction = 1.0):
        self._img_0_input=[]
        self._img_1_input=[]
        self._preprocessor=FeaturesPreprocessor()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
        self._tf_session = tf.compat.v1.Session(config=config)
        with tf.io.gfile.GFile(model_path, 'rb') as graph_file:
            # Load graph off of disk into a graph definition
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(graph_file.read())
            graph_def = optimizeGraph(graph_def,
                                          ['import/dense_2/Sigmoid:0']) # don't optimize this away!
            return_elements = ['input_2:0', 'input_3:0', 'dense_2/Sigmoid:0']
            tensors = tf.import_graph_def(
                graph_def,
                return_elements=return_elements)
            # The first is an input
            self._img_0 = tensors[0]
            self._img_1 = tensors[1]
            # The rest are outputs
            self._distance = tensors[2]
        self._image_shape = self._img_0.shape[1:3]
        print(f"Image shape = {self._image_shape}")
        print(f"Output shape = {self._distance.shape} {self._distance.name}")

    def addPair(self, img0, img1):
        self._img_0_input.append(self._preprocessor(img0, self._image_shape[1], self._image_shape[0]))
        self._img_1_input.append(self._preprocessor(img1, self._image_shape[1], self._image_shape[0]))
        return len(self._img_0_input)

    def process(self):
        result = self._tf_session.run(
            self._distance,
            feed_dict={self._img_0: np.array(self._img_0_input),
                       self._img_1: np.array(self._img_1_input)})
        self._img_0_input = []
        self._img_1_input = []
        return result
