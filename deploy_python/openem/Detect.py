import numpy as np
import cv2

from openem.models import ImageModel
from openem.models import Preprocessor

class SSDDetector(ImageModel):
    preprocessor=Preprocessor(1.0,
                              np.array([-103.939,-116.779,-123.68]),
                              False)
    def addImage(self, image):
        """ Add an image to process in the underlying ImageModel after
            running preprocessing on it specific to this model.

            image: np.array of the underlying image (not pre-processed) to
                   add to the model's current batch.

        """
        return self._addImage(image, self.preprocessor)

    def process(self):
        """ Runs network to find fish in batched images by performing object
            detection with a Single Shot Detector (SSD).

        Returns a list of detections (or None if batch is empty)
        """
        result = super(SSDDetector, self).process()
        if result is None:
            return result


def nms_boxes(bboxes, scores, score_threshold, nms_threshold, top_k = 0):
    """ Performs non-maximum supression on a series of overlapping
        bounding boxes

        bboxes: list of bounding boxes
        score_threshold a threshold used to filter boxes by score.
        nms_threshold a threshold used in non maximum suppression.
        indices the kept indices of bboxes after NMS.
        top_k if `>0`, keep at most @p top_k picked indices.
    """

    # Sort the scores in descending order, create a new list keeping
    # indices and value in a tuple value
    score_sort_list=[]
    for idx,score in enumerate(scores):
        score_sort_list.append((idx,score))

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
