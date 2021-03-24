""" Module for finding ruler masks in raw images """
import numpy as np
import cv2

from openem.models import ImageModel
from openem.models import Preprocessor
from openem.image import crop

class RulerMaskFinder(ImageModel):
    """ Class for finding ruler masks from raw images """
    def __init__(self, model_path, image_dims=None, **kwargs):
        super(RulerMaskFinder,self).__init__(model_path, image_dims, optimize=False, **kwargs)
        self.preprocessor = Preprocessor(1.0 / 128.0, -np.ones(image_dims[-1]), True)
    def addImage(self, image):
        """ Add an image to process in the underlying ImageModel after
            running preprocessing on it specific to this model.

        image: np.ndarray the underlying image (not pre-processed) to add
               to the model's current batch
        """
        return self._addImage(image, self.preprocessor)

    def process(self, postprocess=True):
        """ Runs the base ImageModel and does a high-pass filter only allowing
            matches greater than 127 to make it into the resultant mask

        Returns the mask of the ruler in the size of the network image,
        the user must resize to input image if different.
        """
        model_masks, image_cookies = super(RulerMaskFinder,self).process()
        if model_masks is None:
            return None

        mask_images = []
        num_masks = model_masks.shape[0]
        for idx in range(num_masks):
            # Tensorflow output is 0 to 1
            scaled_image = model_masks[idx] * 255
            if postprocess:
                ret, mask_image = cv2.threshold(scaled_image,
                                                127,
                                                255,
                                                cv2.THRESH_BINARY)
                blurred_image = cv2.medianBlur(mask_image,5)
                mask_images.append(blurred_image)
            else:
                mask_images.append(scaled_image)

        return np.array(mask_images)

def rulerPresent(image_mask):
    """ Returns true if a ruler is present in the frame """
    return cv2.sumElems(image_mask)[0] > 1000.0

def rulerEndpoints(image_mask):
    """
    Find the ruler end points given an image mask
    image_mask: 8-bit single channel image_mask
    """
    image_height = image_mask.shape[0]

    image_mask = image_mask.astype(np.float64)
    image_mask /= 255.0

    # Find center of rotation of the mask
    moments = cv2.moments(image_mask)
    centroid_x = moments['m10'] / moments['m00']
    centroid_y = moments['m01'] / moments['m00']
    centroid = (centroid_x, centroid_y)

    # Find the transofrm to translate the image to the
    # center of the of the ruler
    center_y = image_mask.shape[0] / 2.0
    center_x = image_mask.shape[1] / 2.0
    center = (center_x, center_y)

    diff_x = center_x - centroid_x
    diff_y = center_y - centroid_y

    translation=np.array([[1,0,diff_x],
                         [0,1,diff_y],
                         [0,0,1]])

    min_moment = float('+inf')
    best = None
    best_angle = None
    for angle in np.linspace(-90,90,181):
        rotation = cv2.getRotationMatrix2D(centroid,
                                           float(angle),
                                           1.0)
        # Matrix needs bottom row added
        # Warning: cv2 dimensions are width, height not height, width!
        rotation = np.vstack([rotation, [0,0,1]])
        rt_matrix = np.matmul(translation,rotation)
        rotated = cv2.warpAffine(image_mask,
                                 rt_matrix[0:2],
                                 (image_mask.shape[1],
                                  image_mask.shape[0]))

        rotated_moments = cv2.moments(rotated)
        if rotated_moments['mu02'] < min_moment:
            best_angle = angle
            min_moment = rotated_moments['mu02']
            best = np.copy(rt_matrix)

    #Now that we have the best rotation, find the endpoints
    warped = cv2.warpAffine(image_mask,
                            best[0:2],
                            (image_mask.shape[1],
                             image_mask.shape[0]))

    # Reduce the image down to a 1d line and up convert to 64-bit
    # float between 0 and 1
    col_sum = cv2.reduce(warped,0, cv2.REDUCE_SUM).astype(np.float64)

    # Find the left/right of masked region in the line vector
    # Then, knowing its the center of the transformed image
    # back out the y coordinates in the actual image inversing
    # the transform above
    cumulative_sum = np.cumsum(col_sum[0])
    # Normalize the cumulative sum from 0 to 1
    max_sum=np.max(cumulative_sum)
    cumulative_sum /= max_sum

    # Find the left,right indices based on thresholds
    left_idx = np.searchsorted(cumulative_sum, 0.06, side='left')
    right_idx = np.searchsorted(cumulative_sum, 0.94, side='right')
    width = right_idx - left_idx

    # Add 10% of the ruler width
    left_idx = left_idx-(width*0.10)
    right_idx = right_idx+(width*0.10)

    endpoints=np.array([[[left_idx, image_height / 2],
                         [right_idx, image_height / 2]]])
    # Finally inverse the transform to get the actual y coordinates
    inverse = cv2.invertAffineTransform(best[0:2])
    inverse = np.vstack([inverse, [0,0,1]])
    return cv2.perspectiveTransform(endpoints, inverse)[0]

def rectify(image, endpoints):
    """ Rectifies an image such that the ruler(in endpoints) is flat
        image: array
               Represents an image or image mask
        endpoints: array
                   Represents 2 pair of endpoints for a ruler
    """
    dst = np.array([[image.shape[1]*.1, image.shape[0]/2],
                    [image.shape[1]*.9, image.shape[0]/2]])
    rt_matrix,_ = cv2.estimateAffinePartial2D(endpoints,
                                            dst)
    return cv2.warpAffine(image,
                          rt_matrix,
                          (image.shape[1],image.shape[0]))


def findRoi(image_mask, h_margin):
    """ Returns the roi of a given mask; with additional padding added
        both horizontally and vertically based off of h_margin and the
        underlying aspect ratio.
        image_mask: array
               Represents image mask
        h_margin: int
               Number of pixels to use
    """
    non_zero=cv2.findNonZero(image_mask)
    x,y,w,h = cv2.boundingRect(non_zero)
    aspect = image_mask.shape[0] / image_mask.shape[1]
    v_margin = h_margin * aspect

    # Add margin
    margined_roi=np.zeros(4)
    margined_roi[0] = x - h_margin
    margined_roi[1] = y - v_margin
    margined_roi[2] = w + (h_margin*2)
    margined_roi[3] = h + (v_margin*2)

    # constrain to image dims
    margined_roi[0] = max(margined_roi[0], 0)
    margined_roi[1] = max(margined_roi[1], 0)
    if margined_roi[0] + margined_roi[2] > image_mask.shape[1]:
        margined_roi[2] = image_mask.shape[1] - margined_roi[0]
    if margined_roi[1] + margined_roi[3] > image_mask.shape[0]:
        margined_roi[3] = image_mask.shape[0] - margined_roi[1]

    return (margined_roi[0], margined_roi[1], margined_roi[2],margined_roi[3])


