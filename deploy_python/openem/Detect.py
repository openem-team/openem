from openem.models import ImageModel
from openem.models import Preprocessor

import cv2
import numpy as np
import tensorflow as tf

from collections import namedtuple

Detection=namedtuple('Detection', ['location',
                                   'confidence',
                                   'species'])
class SSDDetector(ImageModel):
    preprocessor=Preprocessor(1.0,
                              np.array([-103.939,-116.779,-123.68]),
                              False)
    _imageSizes = None
    def addImage(self, image):
        """ Add an image to process in the underlying ImageModel after
            running preprocessing on it specific to this model.

            image: np.array of the underlying image (not pre-processed) to
                   add to the model's current batch.

        """
        if self._imageSizes is None:
            self._imageSizes = []
        self._imageSizes.append(image.shape)
        return self._addImage(image, self.preprocessor)

    def process(self):
        """ Runs network to find fish in batched images by performing object
            detection with a Single Shot Detector (SSD).

        Returns a list of Detection (or None if batch is empty)
        """
        batch_result = super(SSDDetector, self).process()
        if batch_result is None:
            return batch_result

        batch_detections=[]
        # Split out the tensor into bounding bboxes per image
        for image_idx,image_result in enumerate(batch_result):
            image_dims=self._imageSizes[image_idx]
            pred_stop = 4
            conf_stop = image_result.shape[1] - 8
            anc_stop = conf_stop + 4
            var_stop = anc_stop + 4
            loc = image_result[:,:pred_stop]
            conf = image_result[:,pred_stop:conf_stop]
            anchors = image_result[:,conf_stop:anc_stop]
            variances = image_result[:,anc_stop:var_stop]
            boxes = decodeBoxes(loc, anchors, variances, image_dims)
            scores=np.zeros(loc.shape[0])
            class_index=np.zeros(loc.shape[0])
            for idx,r in enumerate(conf):
                _,maxScore,__,maxIdx = cv2.minMaxLoc(r[1:])
                scores[idx] = maxScore
                class_index[idx] = maxIdx[1] + 1 # +1 for background class
            indices = tf.image.non_max_suppression(boxes,
                                                   scores,
                                                   200,
                                                   0.01,
                                                   0.45)
            detections = []
            for idx in indices.eval(session=self.tf_session):
                detection = Detection(
                    location = boxes[idx],
                    confidence = scores[idx],
                    species = class_index[idx])
                detections.append(detection)

            def get_confidence(detection):
                return detection.confidence

            detections.sort(key=get_confidence, reverse=True)

            batch_detections.append(detections)
        # Clean up scale factors and return the list
        self._imageSizes = None
        return batch_detections
def decodeBoxes(loc, anchors, variances, img_size):
    """ Decodes bounding box from network output

    loc: Bounding box parameters one box per element
    anchors: Anchors box parameters, one box per element
    variances: Variances per box

    Returns a Nx4 matrix of bounding boxes
    """
    decoded=[]
    image_height = img_size[0]
    image_width = img_size[1]

    anchor_width = anchors[:,2] - anchors[:,0]
    anchor_height = anchors[:,3] - anchors[:,1]

    anchor_center_x = 0.5 * (anchors[:,2] + anchors[:,0])
    anchor_center_y = 0.5 * (anchors[:,3] + anchors[:,1])
    decode_center_x = loc[:,0]*anchor_width*variances[:,0]
    decode_center_x += anchor_center_x
    decode_center_y = loc[:,1]*anchor_height*variances[:,1]
    decode_center_y += anchor_center_y

    decode_width = np.exp(loc[:,2]*variances[:,2])*anchor_width
    decode_height = np.exp(loc[:,3]*variances[:,3])*anchor_height

    decode_x0 = np.maximum((decode_center_x - 0.5 * decode_width) * image_width,0)
    decode_y0 = np.maximum((decode_center_y - 0.5 * decode_height) * image_height,0)
    decode_x1 = np.maximum((decode_center_x + 0.5 * decode_width) * image_width,0)
    decode_y1 = np.maximum((decode_center_y + 0.5 * decode_height) * image_height,0)
    decoded=np.zeros((len(decode_x0),4))
    decoded[:,0] = decode_x0
    decoded[:,1] = decode_y0
    decoded[:,2] = decode_x1 - decode_x0 + 1
    decoded[:,3] = decode_y1 - decode_y0 + 1
    return decoded

"""
#Using tensorflow nms instead --- this one required a function that did
#not work in opencv so well.

def nmsBoxes(bboxes, scores, score_threshold, nms_threshold, top_k = 0):
    \""" Performs non-maximum supression on a series of overlapping
        bounding boxes

        bboxes: list of bounding boxes
        score_threshold a threshold used to filter boxes by score.
        nms_threshold a threshold used in non maximum suppression.
        indices the kept indices of bboxes after NMS.
        top_k if `>0`, keep at most @p top_k picked indices.
    \"""

    # Sort the scores in descending order, create a new list keeping
    # indices and value in a tuple value
    score_sort_list=[]
    for idx,score in enumerate(scores):
        score_sort_list.append((idx,score))

    # inline-function to get second element of tuple
    def second_element(val):
        return val[1]

    # Sort the new list by the 2nd element (score) in descending order
    score_sort_list.sort(key = second_element, reverse = True)

    # If only keeping top_k, truncate the list
    if top_k > 0:
        score_sort_list[:top_k]

    # Iterate over the scores/boxes and pick winners
    winning_indices=[]
    for idx_score in score_sort_list:
        keep = True
        for winner in winning_indices:
            if not keep:
                break
            overlap = 1.0 - cv2.jaccardDistance(idx_score[0], winner)
            keep = overlap <= nms_threshold
        if keep:
            winning_indices.append(idx_score[0])

    return winning_indices
"""
