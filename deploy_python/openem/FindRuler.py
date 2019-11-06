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

    @staticmethod
    def rulerPresent(image_mask):
        """ Returns true if a ruler is present in the frame """
        return cv2.sumElems(image_mask)[0] > 1000.0

    @staticmethod
    def rulerEndpoints(image_mask):
        # Find center of rotation of the mask
        moments = cv2.moments(image_mask)
        centroid_x = moments['m10'] / moments['m00']
        centroid_y = moments['m01'] / moments['m00']
        centroid = np.array([centroid_x, centroid_y])

        # Find the transofrm to translate the image to the
        # center of the of the ruler
        center_y = image_mask.shape[0] / 2.0
        center_x = image_mask.shape[1] / 2.0
        center = np.array([center_x, center_y])

        diff_x = center_x - centroid_x
        diff_y = center_y - centroid_y

        translation=np.array([[1,0,diff_x],
                             [0,1,diff_y],
                             [0,0,1]])

        min_moment = float('+inf')
        best = None
        for angle in range(-90, 90):
            rotation = cv2.getRotationMatrix2D(centroid,
                                               float(angle),
                                               1.0)
            # Matrix needs bottom row added
            rotation = np.vstack(rotation, [0,0,1])
            rt_matrix = translation * rotation
            rotated = cv2.warpAffine(image_mask,
                                     rt_matrix[0:2],
                                     image_mask.shape[0:2])

            rotated_moments = cv2.moments(rotated)
            if moments['mu02'] < min_moment:
                min_moment = moments['mu02']
                best = np.copy(rt_matrix)

        #Now that we have the best rotation, find the endpoints
