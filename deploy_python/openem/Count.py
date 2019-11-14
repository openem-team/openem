""" Module for finding keyframes """
import tensorflow as tf
import numpy as np
import cv2

from openem.models import ImageModel
from openem.models import Preprocessor
from openem.image import crop

KEYFRAME_OFFSET = 32
MIN_SPACING = 1
PEAK_THRESHOLD = 0.03
AREA_THRESHOLD = 0.10

class KeyframeFinder:
    def __init__(self, model_path, img_width, img_height, gpu_fraction=1.0):
        """ Initialize a keyframe finder model. Tracks an object detection
            across frames. In other words associates object detections of the
            same object across frames of a video.

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

        self.img_width = img_width
        self.img_height = img_height
    def normalizeDetection(self, detection):
        print(detection)
        """ Normalize a detection coordinates to be relative coordinates """
        return np.array([detection[0] / self.img_width,
                         detection[1] / self.img_height,
                         detection[2] / self.img_width,
                         detection[3] / self.img_height])

    def process(self, classifications, detections):
        """ Process the list of classifications and detections, which
            must be the same length.

            The outer dimension in each parameter is a video; and the inner 
            each detection in a given video

            classifications: list of list of openem.Classify.Classfication
            detections: list of list of openem.Detect.Detection
        """
        det_len = len(detections)
        if len(classifications) != len(detections):
            raise Exception("Classifications / Detections difer in length!")

        # Convert classifications and detections to input required for network
        seq_len = int(self.input_tensor.shape[1])
        fea_len = int(self.input_tensor.shape[2])
        print(f"shape = {self.input_tensor.shape}")
        input_data = np.zeros((seq_len,fea_len))

        # Add padding before and after sequence based on KEYFRAME_OFFSET
        input_data[:KEYFRAME_OFFSET,0] = np.ones(KEYFRAME_OFFSET)
        input_data[det_len:det_len+KEYFRAME_OFFSET,0] = np.ones(KEYFRAME_OFFSET)
        # Iterate through each frame of the data
        for idx, detection in enumerate(detections):
            classification = classifications[idx]

            # We have already padded before and after
            seq_idx = idx + KEYFRAME_OFFSET
            # Skip through frames with no detections
            if len(detection) == 0:
                input_data[seq_idx][0] = 1.0
                continue

            # Do a size check on input
            # We expect either 1 or 2 models per sequence
            num_species = len(classification.species)
            num_cover = len(classification.cover)
            num_loc = len(detection.location)
            num_fea = num_species + num_cover + num_loc + 2
            num_of_models = int(fea_len / num_fea)

            if num_of_models != 2 and num_of_models != 1:
                raise Exception('Bad Feature Length')

            # Layout of the feature is:
            # Species, Cover, Normalized Location, Confidence, SSD Species
            # Optional duplicate
            
            for model_idx in range(num_of_models):
                # Calculate indices of vector based on model_idx
                fea_idx = model_idx * num_fea
                species_stop = fea_idx + num_species
                cover_stop = species_stop + num_cover
                loc_stop = cover_stop + num_loc
                ssd_conf = loc_stop
                ssd_species = ssd_conf + 1
                
                input_data[seq_idx,fea_idx:species_stop] = \
                    classification.species
                input_data[seq_idx,species_stop:cover_stop] = \
                    classification.cover
                input_data[seq_idx,cover_stop:loc_stop] = \
                    self.normalizeDetection(detection.location)
                input_data[seq_idx, ssd_conf] = detection.confidence
                input_data[seq_idx, ssd_species] = detection.species
                
        result = self.tf_session.run(self.output_tensor,
                                     feed_dict={self.input_tensor:
                                                np.array([input_data])})
        
