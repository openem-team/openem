import os
import pickle
import functools
import glob
import cv2
import tables
import numpy as np
from collections import OrderedDict
from keras.models import model_from_yaml
from keras.models import load_model
from keras.models import Model
from keras.layers import concatenate
from keras.layers import Input
from extractors import AppearanceFeatureExtractor
from extractors import AppearanceFeatureNormalizer
from extractors import MotionFeatureExtractor
from extractors import InitFeatureExtractor
from providers import DataProvider
from providers import SequenceProvider
from providers import TrackletProvider
from providers import CountProvider
from providers import ClassificationProvider
from utilities import TrackData
from utilities import CountData
from utilities import ClassData
from utilities import get_image
from natsort import natsorted

class ModelData:
    ok_features = ["cnn", "appearance", "motion", "target", "classification"]

    """ Interface to directory containing model data.
    """
    def __init__(self, data_dir):
        """ Constructor.
            Inputs:
            data_dir -- Path to model data directory.
        """
        ## Base path to data directory.
        self.data_dir = data_dir.strip()

    def track_dirs(self):
        """ Returns list of track directories use to create this model.
        """
        path = os.path.join(self.data_dir, "track_dirs.txt")
        f = open(path, "r")
        dirs = f.readlines()
        dirs = [d.strip() for d in dirs]
        return dirs

    def save_track_dirs(self, track_dirs):
        """ Saves list of track directories to text file.
        """
        path = os.path.join(self.data_dir, "track_dirs.txt")
        f = open(path, "w")
        for track_dir in track_dirs:
            f.write("{}\n".format(track_dir))
        f.close()

    def mean_image(self):
        """ Returns mean image.
            Will use {data_dir}/mean_image.png if supplied, else
            will use imagenet means.
        """
        try:
            path = os.path.join(self.data_dir, "mean_image.png")
            return cv2.imread(path).astype(np.float)
        except:
            # This is in BGR
            return np.array([[103.939, 116.779, 123.68 ]])

    def save_mean_image(self, mean_img):
        """ Saves mean image to file.
        """
        path = os.path.join(self.data_dir, "mean_image.png")
        cv2.imwrite(path, mean_img)

    def cnn_training(self):
        """ Returns CNN training data.
        """
        return np.load(os.path.join(self.data_dir, "cnn_training_data.npz"))

    def save_cnn_training(self, index, outputs):
        """ Saves CNN training data.
            Inputs:
            index -- 2xN array of index pairs.
            outputs -- N array of outputs.
        """
        np.savez(os.path.join(self.data_dir, "cnn_training_data.npz"),
            index=index,
            outputs=outputs)

    def cnn_validate(self):
        """ Returns CNN validate data.
        """
        return np.load(os.path.join(self.data_dir, "cnn_validate_data.npz"))

    def save_cnn_validate(self, index, outputs):
        """ Saves CNN validate data.
            Inputs:
            index -- 2xN array of index pairs.
            outputs -- N array of outputs.
        """
        np.savez(os.path.join(self.data_dir, "cnn_validate_data.npz"),
            index=index,
            outputs=outputs)

    def cnn_checkpoints_dir(self):
        return os.path.join(self.data_dir, 'cnn_checkpoints')

    def cnn_checkpoint_periodic(self):
        return os.path.join(
            self.cnn_checkpoints_dir(),
            'periodic-{epoch:03d}-{val_loss:.4f}.hdf5'
        )

    def cnn_checkpoint_best(self):
        return os.path.join(
            self.cnn_checkpoints_dir(),
            'best-{epoch:03d}-{val_loss:.4f}.hdf5'
        )

    def save_weights(self, feature, epoch, model):
        if not feature in self.ok_features:
            raise ValueError("Invalid feature {}!".format(feature))
        fname = feature + "_weights_epoch{:02d}.h5".format(epoch)
        path = os.path.join(self.data_dir, fname)
        model.save_weights(path)

    def save_architecture(self, feature, model):
        if not feature in self.ok_features:
            raise ValueError("Invalid feature {}!".format(feature))
        path = os.path.join(self.data_dir, feature + "_architecture.yaml")
        arch_file = open(path, "w")
        arch_file.write(model.to_yaml())
        arch_file.close()

    def training_loss(self, feature, epoch):
        """ Returns CNN training loss for given feature and epoch.
        """
        if not feature in self.ok_features:
            raise ValueError("Invalid feature {}!".format(feature))
        fname = feature + "_training_loss_epoch{:02d}.npy".format(epoch)
        path = os.path.join(self.data_dir, fname)
        if os.path.exists(path):
            return np.load(path)
        else:
            return None

    def save_training_loss(self, feature, epoch, training_loss):
        """ Saves training loss for given feature and epoch.
        """
        if not feature in self.ok_features:
            raise ValueError("Invalid feature {}!".format(feature))
        fname = feature + "_training_loss_epoch{:02d}".format(epoch)
        path = os.path.join(self.data_dir, fname)
        np.save(path, training_loss)

    def validate_loss(self, feature, epoch):
        """ Returns CNN validate loss for given feature and epoch.
        """
        if not feature in self.ok_features:
            raise ValueError("Invalid feature {}!".format(feature))
        fname = feature + "_validate_loss_epoch{:02d}.npy".format(epoch)
        path = os.path.join(self.data_dir, fname)
        if os.path.exists(path):
            return np.load(path)
        else:
            return None

    def save_validate_loss(self, feature, epoch, validate_loss):
        if not feature in self.ok_features:
            raise ValueError("Invalid feature {}!".format(feature))
        fname = feature + "_validate_loss_epoch{:02d}".format(epoch)
        path = os.path.join(self.data_dir, fname)
        np.save(path, validate_loss)

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

    def appearance_extractor(self):
        """ Returns appearance feature extractor.
        """
        best = glob.glob(os.path.join(self.cnn_checkpoints_dir(), 'best*.hdf5'))
        latest = max(best, key=os.path.getctime)
        model = load_model(latest)
        cnn = Model(
            inputs=model.get_layer("model_1").get_input_at(0),
            outputs=model.get_layer("model_1").get_output_at(0))
        cnn.summary()
        mean_img = self.mean_image()
        return AppearanceFeatureExtractor(cnn, mean_img)

    def appearance_comparator(self):
        """ Returns appearance comparison network.
        """
        best = glob.glob(os.path.join(self.cnn_checkpoints_dir(), 'best*.hdf5'))
        latest = max(best, key=os.path.getctime)
        model = load_model(latest)
        trk_fea = Input(shape=(500,))
        det_fea = Input(shape=(500,))
        model.summary()
        out = model.get_layer("lambda_1")([trk_fea, det_fea])
        out = model.get_layer("dense_4")(out)
        comparator = Model(inputs=[trk_fea, det_fea], outputs=out)
        comparator.summary()
        return comparator

    def appearance_normalizer(self):
        """ Returns appearance feature normalizer.
        """
        stats = self.feature_stats()["cnn_stats"]
        return AppearanceFeatureNormalizer(stats)

    def motion_extractor(self):
        """ Returns motion feature extractor.
        """
        stats = self.feature_stats()["vel_stats"]
        return MotionFeatureExtractor(stats)

    def init_extractor(self, vid_h, vid_w):
        """ Returns init extractor.
            Inputs:
            vid_h -- Height of video.
            vid_w -- Width of video.
        """
        stats = self.feature_stats()["init_stats"]
        return InitFeatureExtractor(stats, vid_h, vid_w)

    def init_svm(self):
        """ Returns init svm.
        """
        path = os.path.join(self.data_dir, "init_svm.pkl")
        f = open(path, "rb")
        return pickle.load(f)

    def save_init_svm(self, svm):
        """ Saves init svm.
        """
        path = os.path.join(self.data_dir, "init_svm.pkl")
        f = open(path, "wb")
        pickle.dump(svm, f)

    def data_providers(
        self,
        augs,
        batch_size,
        train_queue,
        validate_queue,
        do_plots):
        """ Returns data providers.
            Inputs:
            augs -- List of augmentation functions.
            batch_size -- Batch size.
            train_queue -- Queue to push training data into.
            validate_queue -- Queue to push validation data into.
            do_plots -- If true, provider will do plotting.
        """
        # Create the index mapping and cumulative indices.
        dir_idx_list = []
        idx_list = []
        tdata_list = [TrackData(os.path.join(self.data_dir,d)) for d in self.track_dirs()]
        for dir_idx, track_data in enumerate(tdata_list):
            num_img = track_data.num_detection_images()
            dir_idx_list += [dir_idx for _ in range(num_img)]
            idx_list += range(num_img)
        mapping = np.transpose([dir_idx_list, idx_list])
        # Bind the track data list and index mapping to image accessor.
        get_img_bound = functools.partial(get_image, tdata_list, mapping)
        return (
            DataProvider(
                self.cnn_training(),
                get_img_bound,
                self.mean_image(),
                augs,
                batch_size,
                train_queue,
                do_plots),
            DataProvider(
                self.cnn_validate(),
                get_img_bound,
                self.mean_image(),
                augs,
                batch_size,
                validate_queue,
                do_plots))

    def sequence_providers(
        self,
        trk_list,
        assoc_examples,
        validate_ratio,
        timesteps,
        batch_size,
        feature,
        do_plots):
        """ Returns sequence providers.
            Inputs:
            trk_list -- List of lists of track objects (one list per
                directory).
            assoc_examples -- List of lists, each containing directory
                index, track id, sequence length, detection, and output.
            validate_ratio -- Fraction of data to set aside for validation.
            timesteps -- Number of timesteps in sequence.
            batch_size -- Batch size.
            feature -- One of 'appearance', 'motion', or 'target'.
            max_drop_fraction -- Maximum fraction of time steps that can
                be dropped from a sequence.  Only applied to training
                sequence generator.
            do_plots -- If true, provider will do plotting.
        """
        validate_stop = int(len(assoc_examples) * validate_ratio)
        if feature == "appearance":
            extractor = {"appearance" : self.appearance_normalizer()}
        elif feature == "motion":
            extractor = {"motion" : self.motion_extractor()}
        elif feature == "target":
            extractor = OrderedDict([
                ("appearance", self.appearance_normalizer()),
                ("motion", self.motion_extractor())])
        return (
            SequenceProvider(
                trk_list,
                assoc_examples[validate_stop:],
                timesteps,
                batch_size,
                extractor,
                do_plots,
                self),
            SequenceProvider(
                trk_list,
                assoc_examples[:validate_stop],
                timesteps,
                batch_size,
                extractor,
                do_plots,
                self))

    def model(self, feature, include_last=False, epoch=None):
        """ Constructs a model associated with the specified feature.
            Layers of the model are renamed to include the feature name.
            Inputs:
            feature -- Which type of model to return.  Must be one of
                'appearance', 'motion', 'target', or 'classification'.
            include_last -- If true, include the last single output layer.
            epoch -- If given, the epoch to load.  Otherwise will use the
                last epoch.
        """
        arch_file = "{}_architecture.yaml".format(feature)
        if epoch is None:
            weights_file = natsorted([f for f in os.listdir(self.data_dir)
                if "{}_weights".format(feature) in f])[-1]
        else:
            weights_file = "{}_weights_epoch{:03d}.h5".format(feature, epoch)
        arch_path = os.path.join(self.data_dir, arch_file)
        arch_fp = open(arch_path, "r")
        weights_path = os.path.join(self.data_dir, weights_file)
        model = model_from_yaml(arch_fp.read())
        arch_fp.close()
        model.load_weights(weights_path)
        for i, a in enumerate(range(len(model.layers))):
            model.layers[a].name = model.layers[a].name + feature
        if include_last:
            return model
        else:
            return Model(
                inputs=model.input,
                outputs=model.layers[-3].output)

    def tracklet_checkpoints_dir(self):
        return os.path.join(self.data_dir, 'tracklet_checkpoints')

    def tracklet_checkpoint_periodic(self):
        return os.path.join(
            self.tracklet_checkpoints_dir(),
            'periodic-{epoch:03d}-{val_loss:.4f}.hdf5'
        )

    def tracklet_checkpoint_best(self):
        return os.path.join(
            self.tracklet_checkpoints_dir(),
            'best-{epoch:03d}-{val_loss:.4f}.hdf5'
        )

    def tracklet_providers(self, batch_size, timesteps, fold):
        assert((fold > 0) and (fold <= 10))
        track_data_list = [TrackData(t) for t in self.track_dirs()]
        chunk_size = len(track_data_list) / 10
        val_start = int((fold-1) * chunk_size)
        val_stop = int(fold * chunk_size)
        train_data = track_data_list[:val_start] + track_data_list[val_stop:]
        val_data = track_data_list[val_start:val_stop]
        train_provider = TrackletProvider(train_data, batch_size, timesteps)
        val_provider = TrackletProvider(val_data, batch_size, timesteps)
        return (train_provider, val_provider)

    def tracklet_comparator(self):
        """Returns tracklet comparator model.
        """
        best = glob.glob(os.path.join(self.tracklet_checkpoints_dir(), 'best*.hdf5'))
        latest = max(best, key=os.path.getctime)
        model = load_model(latest)
        model.summary()
        return model

    def count_checkpoints_dir(self):
        return os.path.join(self.data_dir, 'count_checkpoints')

    def count_checkpoint_periodic(self):
        return os.path.join(
            self.count_checkpoints_dir(),
            'periodic-{epoch:03d}-{val_loss:.4f}.hdf5'
        )

    def count_checkpoint_best(self):
        return os.path.join(
            self.count_checkpoints_dir(),
            'best-{epoch:03d}-{val_loss:.4f}.hdf5'
        )

    def count_providers(self, count_dir, batch_size, fold):
        assert((fold > 0) and (fold <= 10))
        train_folds = list(range(0, fold)) + list(range((fold+1),10))
        test_folds = [fold,]
        count_data = CountData(count_dir)
        train_provider = CountProvider(count_data, train_folds, batch_size)
        val_provider = CountProvider(count_data, test_folds, batch_size)
        return (train_provider, val_provider)

    def count_model(self):
        """Returns count model.
        """
        best = glob.glob(os.path.join(self.count_checkpoints_dir(), 'best*.hdf5'))
        latest = max(best, key=os.path.getctime)
        model = load_model(latest)
        return model

    def classification_checkpoints_dir(self):
        return os.path.join(self.data_dir, 'classification_checkpoints')

    def classification_checkpoint_periodic(self):
        return os.path.join(
            self.classification_checkpoints_dir(),
            'periodic-{epoch:03d}-{val_loss:.4f}.hdf5'
        )

    def classification_checkpoint_best(self):
        return os.path.join(
            self.classification_checkpoints_dir(),
            'best-{epoch:03d}-{val_loss:.4f}.hdf5'
        )

    def classification_providers(self, class_dir, batch_size, fold):
        assert((fold > 0) and (fold <= 10))
        train_folds = list(range(0, fold)) + list(range((fold+1),10))
        test_folds = [fold,]
        class_data = ClassData(class_dir)
        train_provider = ClassificationProvider(class_data, train_folds, batch_size)
        val_provider = ClassificationProvider(class_data, test_folds, batch_size)
        return (train_provider, val_provider)

    def classification_model(self):
        """Returns classification model.
        """
        path = os.path.join(
            self.classification_checkpoints_dir(),
            'best*.hdf5'
        )
        best = glob.glob(path)
        latest = max(best, key=os.path.getctime)
        model = load_model(latest)
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
