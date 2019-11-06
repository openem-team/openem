""" Module for finding ruler masks in raw images """
import numpy as np
import cv2

from openem.models import ImageModel
from openem.models import Preprocessor

class RulerMaskFinder(ImageModel):
    """ Class for finding ruler masks from raw images """
    preprocessor=Preprocessor(1.0/128.0,
                              np.array([-1,-1,-1]),
                              True)

    def addImage(self, image):
        """ Add an image to process in the underlying ImageModel after
            running preprocessing on it specific to this model.

        image: np.ndarray the underlying image (not pre-processed) to add
               to the model's current batch
        """
        return self._addImage(image, self.preprocessor)

    def process(self):
        """ Runs the base ImageModel and does a high-pass filter only allowing
            matches greater than 127 to make it into the resultant mask

        Returns the mask of the ruler in the size of the network image,
        the user must resize to input image if different.
        """
        model_masks = super(RulerMaskFinder,self).process()

        mask_images = []
        num_masks = model_masks.shape[0]
        for idx in range(num_masks):
            # Tensorflow output is 0 to 1
            scaled_image = model_masks[idx] * 255
            ret, mask_image = cv2.threshold(scaled_image,
                                            127,
                                            255,
                                            cv2.THRESH_BINARY)
            blurred_image = cv2.medianBlur(mask_image,5)
            mask_images.append(blurred_image)

        return np.array(mask_images)
