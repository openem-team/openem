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

def peak_sum(array, idx, width):
    sum_value = array[idx]
    if idx - width > 0:
        sum_value += array[idx-width]
    if idx + width < len(array):
        sum_value += array[idx+width]
    return sum_value

class KeyframeFinder:
    def __init__(self, model_path, img_width, img_height, gpu_fraction=1.0):
        """ Initialize a keyframe finder model. Gives a list of keyframes for
            each species. Caveats of this model:

            - Assumes tracking 1 classification/detection per frame

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

    def process(self, classifications, detections):
        """ Process the list of classifications and detections, which
            must be the same length.

            The outer dimension in each parameter is a frame; and the inner
            a list of detection or classification in a given frame

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
        for idx, frame_detections in enumerate(detections):
            # We have already padded before and after
            seq_idx = idx + KEYFRAME_OFFSET

            # Skip through frames with no detections
            if len(frame_detections) == 0:
                input_data[seq_idx][0] = 1.0
                continue

            detection = frame_detections[0]
            classification = classifications[idx][0]

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
                    self._normalizeDetection(detection.location)
                input_data[seq_idx, ssd_conf] = detection.confidence
                input_data[seq_idx, ssd_species] = detection.species

        result = self.tf_session.run(self.output_tensor,
                                     feed_dict={self.input_tensor:
                                                np.array([input_data])})

        keyframes=[]
        output_length = seq_len - (2*KEYFRAME_OFFSET)
        assert output_length == result.shape[1]

        # Iterate over the output and if we found a match, add it to the
        # keyframe lists

        array = result[0]

        while True:
            max_idx = np.argmax(array)
            max_value = array[max_idx]
            if max_value < PEAK_THRESHOLD:
                return keyframes

            area_sum = peak_sum(array, max_idx, MIN_SPACING)
            low_idx = max(max_idx-MIN_SPACING,0)
            limit = min(max_idx+MIN_SPACING+1,len(array))
            if area_sum > AREA_THRESHOLD:
                max_clear = 0.0
                clear_idx = None
                for area_idx in range(low_idx,limit):
                    class_idx = area_idx - KEYFRAME_OFFSET
                    if len(classifications[class_idx]) == 0:
                        continue
                    element_cover = classifications[class_idx][0].cover[2]
                    if element_cover > max_clear:
                        max_clear = element_cover
                        clear_idx = area_idx

                if clear_idx is not None:
                    keyframes.append(clear_idx)
                    keyframes.sort()

            # Zero out the area identified
            for clear_idx in range(low_idx, limit):
                array[clear_idx] = 0.0
