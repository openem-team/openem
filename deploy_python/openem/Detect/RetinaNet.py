
import tensorflow as tf
import numpy as np
from openem.models import ImageModel

import cv2

from openem.Detect import Detection

class SubtractMeanImage:
    """ Subtract a mean image from the input
        Meets the callable interface of openem.Detect.Preprocessor
    """
    def __init__(self,meanImage):
        self.mean_image = meanImage

    def __call__(self, image, requiredWidth, requiredHeight):
        #Reverse color channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        resized_image = cv2.resize(image, (requiredWidth, requiredHeight))
        if self.mean_image:
            for dim in [0,1,2]:
                resized_image[:,:,dim] -= self.mean_image[:,:,dim]
        return resized_image

NETWORK_IMAGE_SHAPE=(720,1280)
class RetinaNetDetector(ImageModel):
    def __init__(self, modelPath, meanImage, gpuFraction=1.0):
        """ Initialize the RetinaNet Detector model
        modelPath: str
                   path-like object to frozen pb graph
        meanImage: np.array
                   Mean image subtracted from image prior to network
                   insertion. Can be None.
        """
        super(RetinaNetDetector,self).__init__(modelPath,
                                               gpuFraction,
                                               'input_1:0',
                                               'nms/map/TensorArrayStack/TensorArrayGatherV3:0')
        self.input_shape[1:3] = NETWORK_IMAGE_SHAPE

        if meanImage:
            resized_mean = cv2.resize(meanImage,(NETWORK_IMAGE_SHAPE[1],
                                                 NETWORK_IMAGE_SHAPE[0]))
            self.preprocessor=SubtractMeanImage(resized_mean)
        else:
            self.preprocessor=SubtractMeanImage(None)

        self._imageSizes = None

    def addImage(self, image):
        if self._imageSizes is None:
            self._imageSizes = []
        self._imageSizes.append(image.shape)
        return super(RetinaNetDetector, self)._addImage(image,
                                                        self.preprocessor)

    def process(self, threshold=0.0, **kwargs):
        detections = super(RetinaNetDetector,self).process()

        # clip to image shape
        detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
        detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
        detections[:, :, 2] = np.minimum(NETWORK_IMAGE_SHAPE[1], detections[:, :, 2])
        detections[:, :, 3] = np.minimum(NETWORK_IMAGE_SHAPE[0], detections[:, :, 3])

        num_images = detections.shape[0]
        for idx in range(num_images):
            # correct boxes for image scale
            h_scale = NETWORK_IMAGE_SHAPE[0] / self._imageSizes[idx][0]
            w_scale = NETWORK_IMAGE_SHAPE[1] / self._imageSizes[idx][1]

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
            if frame:
                this_frame = frame + img_idx
            else:
                this_frame = None

            image_detections=[]
            for detection in detections[img_idx, ...]:
                label = np.argmax(detection[4:])
                confidence = float(detection[4 + label])
                if confidence > threshold:
                    detection = Detection(location=detection[:4].tolist(),
                                          confidence=confidence,
                                          species=float(label),
                                          frame=this_frame,
                                          video_id=video_id)
                    image_detections.append(detection)
            results.append(image_detections)

        self._imageSizes = None
        return results
