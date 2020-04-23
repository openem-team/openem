import numpy as np

class MotionFeatureExtractor:
    """ Functor for extracting motion features from a detection.
    """
    def __init__(self, stats):
        """ Constructor.
        """
        ## Dict containing mean and std of motion features.
        self.stats = stats
    
    def __call__(self, det, last=None):
        """ Extracts motion features from a detection.
            Inputs:
            det -- Current detection.
            last -- Previous detection.
        """
        width = float(det["w"])
        height = float(det["h"])
        if last is None:
            features = np.array([0.0, 0.0, width, height])
        else:
            x_diff = float(det["center_x"]) - float(last["center_x"])
            y_diff = float(det["center_y"]) - float(last["center_y"])
            frame_diff = float(det["frame"]) - float(last["frame"])
            if frame_diff == 0.0:
                features = np.array([0.0, 0.0, width, height])
            else:
                features = np.array([
                    x_diff / frame_diff, 
                    y_diff / frame_diff,
                    width,
                    height])
        features = features - self.stats["mean"]
        features = np.divide(features, self.stats["std"])
        return features
        
