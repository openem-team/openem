""" RetinaNet Object Detector for OpenEM """

import tensorflow as tf
import numpy as np
from openem.models import ImageModel

import cv2
import math

from openem.Detect import Detection
from openem.image import force_aspect

class RetinaNetPreprocessor:
    """ Perform preprocessinig for RetinaNet inputs
        Meets the callable interface of openem.Detect.Preprocessor
    """
    def __init__(self,meanImage=None):
        self.mean_image = meanImage

    def __call__(self, image, requiredWidth, requiredHeight):
        #TODO: (Provide way to optionally convert channel ordering?)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)

        required_aspect = requiredWidth / requiredHeight
        image = force_aspect(image, required_aspect)
        resized_image = cv2.resize(image, (requiredWidth, requiredHeight))
        if self.mean_image:
            for dim in [0,1,2]:
                resized_image[:,:,dim] -= self.mean_image[:,:,dim]
        else:
            # Use the ImageNet mean image by default; which in BGR is:
            imagenet_mean = np.array([103.939, 116.779, 123.68 ])
            resized_image -= imagenet_mean
        return resized_image

class RetinaNetDetector(ImageModel):
    def __init__(self, modelPath, meanImage=None, gpuFraction=1.0, imageShape=(360,720)):
        """ Initialize the RetinaNet Detector model
        modelPath: str
                   path-like object to frozen pb graph
        meanImage: np.array
                   Mean image subtracted from image prior to network
                   insertion. Can be None.
        image_shape: tuple
                   (height, width) of the image to feed into the detector network.
        """
        super(RetinaNetDetector,self).__init__(modelPath,
                                               gpuFraction,
                                               'input_1:0',
                                               'nms/map/TensorArrayStack/TensorArrayGatherV3:0')
        self.input_shape[1:3] = imageShape

        self.image_shape = imageShape
        self.network_aspect = imageShape[1] / imageShape[0]

        if meanImage:
            resized_mean = cv2.resize(meanImage,(imageShape[1],
                                                 imageShape[0]))
            self.preprocessor=RetinaNetPreprocesser(meanImage=resized_mean)
        else:
            self.preprocessor=RetinaNetPreprocessor(meanImage=None)

        self._imageSizes = None

    def addImage(self, image):
        if self._imageSizes is None:
            self._imageSizes = []

        # Determine the actual shape of the image as it goes into the network
        # To account for padding to aspect ratio
        img_height = image.shape[0]
        img_width = image.shape[1]
        img_aspect = img_width / img_height
        if math.isclose(img_aspect, self.network_aspect):
            self._imageSizes.append(image.shape)
        elif img_aspect < self.network_aspect:
            #Image is boxer than we want
            new_width = round(img_height * self.network_aspect)
            self._imageSizes.append((img_height, new_width))
        else:
            new_height = round(img_width / self.network_aspect)
            self._imageSizes.append((new_height,img_width))
        return super(RetinaNetDetector, self)._addImage(image,
                                                        self.preprocessor)

    def process(self, threshold=0.0, **kwargs):
        detections = super(RetinaNetDetector,self).process()

        # clip to image shape
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(self.image_shape[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(self.image_shape[0], detections[:, :, 3])

        num_images = detections.shape[0]
        for idx in range(num_images):
            # correct boxes for image scale
            # Keep in mind there is a shift here potentially to force
            # an aspect ratio.
            h_scale = self.image_shape[0] / self._imageSizes[idx][0]
            w_scale = self.image_shape[1] / self._imageSizes[idx][1]

            detections[idx, :, 0] /= w_scale
            detections[idx, :, 1] /= h_scale
            detections[idx, :, 2] /= w_scale
            detections[idx, :, 3] /= h_scale


        # change to (x, y, w, h) (MS COCO standard)
        detections[:, :, 2] -= detections[:, :, 0]
        detections[:, :, 3] -= detections[:, :, 1]

        results=[]
        frame = kwargs.get('frame', None)
        video_id = kwargs.get('video_id', None)
        # compute predicted labels and scores
        num_imgs = detections.shape[0]
        for img_idx in range(num_imgs):
            if not frame is None:
                this_frame = frame + img_idx
            else:
                this_frame = None

            image_detections=[]
            for detection in detections[img_idx, ...]:
                label = int(detection[4])
                max_confidence = detection[5+label])
                if max_confidence >= threshold:
                    detection = Detection(location=detection[:4].tolist(),
                                          confidence=detection[5:].tolist(),
                                          # OpenEM uses 1-based indexing
                                          species=label+1,
                                          frame=this_frame,
                                          video_id=video_id)
                    image_detections.append(detection)
            results.append(image_detections)

        self._imageSizes = None
        return results
