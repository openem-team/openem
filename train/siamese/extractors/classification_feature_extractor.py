import cv2
import numpy as np

class ClassificationFeatureExtractor:
    """ Functor for extraction of classification features from detections.
    """
    def __init__(self, model, mean_img):
        """ Constructor.
            Inputs:
            model -- Model used to extract appearance features.
            mean_img -- Mean image which is subtracted before 
                applying the model.
        """
        ## Model used to extract appearance features.
        self.model = model
        ## Mean image which is subtracted before applying the model.
        self.mean_img = mean_img
        ## Image shape.
        self.shape = tuple(self.model.input.shape[1:3].as_list())

    def __call__(self, img):
        """ Extracts features from the given image.
            Inputs:
            img -- Image associated with a detection.
            Returns:
            Appearance feature vector.
        """
        img = np.flip(img, axis=2) # Go to RGB
        img = img.astype(np.float)
        img = cv2.resize(img, self.shape)
        img = np.expand_dims(img, axis=0)
        features = self.model.predict(img - self.mean_img)
        return features

class ClassificationFeatureNormalizer:
    """ Functor for normalizing appearance features.
    """
    def __init__(self, stats):
        ## Dict containing mean and std of appearance features.
        self.stats = stats

    def __call__(self, fea):
        features = np.squeeze(fea) - self.stats["mean"]
        features = np.divide(features, self.stats["std"] + 1e-8)
        return features
