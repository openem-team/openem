import os
import shutil
import glob
import pickle
import cv2
import numpy as np
from keras.models import model_from_yaml
from keras.models import Model
from extractors import ClassificationFeatureExtractor
from extractors import ClassificationFeatureNormalizer
from utilities import ensure_path_exists

class DetidData:
    """ Interface to directory containing detection id data.
    """

    def __init__(self, data_dir):
        """ Constructor.
            Inputs:
            data_dir -- Path to detid data directory.
        """
        ## Base path to data directory.
        self.data_dir = data_dir.strip()

    def model(self, epoch, include_softmax=True):
        """ Returns the model.
            Inputs:
            epoch -- Epoch for which weights should be loaded.
        """
        arch_path = os.path.join(self.data_dir, "cnn_architecture.yaml")
        model_f = open(arch_path, "r")
        model = model_from_yaml(model_f.read())
        weights_path = os.path.join(self.data_dir,
            "cnn_weights_epoch{:02d}.h5".format(epoch))
        model.load_weights(weights_path)
        if not include_softmax:
            model = Model(
                inputs=model.input, 
                outputs=model.layers[-2].output)
        model.summary()
        return model

    def species_names(self):
        """ Returns a list of species names corresponding to 
            model output indices.
        """
        names_path = os.path.join(self.data_dir, "species_names.txt")
        names_f = open(names_path, "r")
        names = names_f.read()
        names = names.split(",")
        names = [n.strip() for n in names]
        return names

    def save_species_names(self, names):
        """ Saves off a list of species names.
        """
        names_path = os.path.join(self.data_dir, "species_names.txt")
        names_f = open(names_path, "w")
        names_f.write(",".join(names))
        
    def mean_image(self):
        """ Returns mean image.
        """
        path = os.path.join(self.data_dir, "mean_img.png")
        img = cv2.imread(path).astype(np.float)
        img = np.flip(img, axis=2)
        return img

    def feature_stats(self):
        """ Opens and reads feature stats file.  Returns unpickled file.
        """
        stats_path = os.path.join(self.data_dir, "feature_stats.pkl")
        stats_file = open(stats_path, "rb")
        return pickle.load(stats_file)

    def save_feature_stats(self, stats):
        """ Saves feature stats to file.
        """
        stats_path = os.path.join(self.data_dir, "feature_stats.pkl")
        stats_file = open(stats_path, "wb")
        pickle.dump(stats, stats_file)

    def extractor(self, epoch):
        """ Returns feature extractor object.
        """
        return ClassificationFeatureExtractor(
            self.model(epoch, False), 
            self.mean_image())

    def normalizer(self):
        """ Returns feature normalizer.
        """
        return ClassificationFeatureNormalizer(self.feature_stats())
