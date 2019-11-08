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

        
