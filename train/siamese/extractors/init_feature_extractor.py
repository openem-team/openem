import numpy as np

class InitFeatureExtractor:
    """ Functor for extracting init features from a detection.
    """
    def __init__(self, stats, vid_h, vid_w):
        """ Constructor.
            Inputs:
            stats -- Feature stats dict.
            vid_h -- Height of video.
            vid_w -- Width of video.
        """
        ## Dict containing mean and std of motion features.
        self.stats = stats
        ## Height of source video.
        self.vid_h = vid_h
        ## Width of source video.
        self.vid_w = vid_w
    
    def __call__(self, det):
        """ Extracts init features from a detection.
            Inputs:
            det -- Current detection.
        """
        root_area = np.sqrt(float(det["w"]) * float(det["h"]))
        dist = float(det["center_x"])
        dist = min(self.vid_w - float(det["center_x"]), dist)
        dist = min(float(det["center_y"]), dist)
        dist = min(self.vid_h - float(det["center_y"]), dist)
        prob = float(det["prob"])
        features = np.array([root_area, dist, prob])
        features = features - self.stats["mean"]
        features = np.divide(features, self.stats["std"])
        return features
        
