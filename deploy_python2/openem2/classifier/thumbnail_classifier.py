"""
Thumbnail classification implementation
"""
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, applications, models, backend, experimental
import tensorflow_hub as hub

class Dropout_MC(tf.keras.layers.Layer):
    """ Custom Dropout layer to support randomization during inference
    TODO: Fixme
    """
    def __init__(self, rate, **kwargs):
        super(Dropout_MC, self).__init__(**kwargs)
        self.rate = rate

    def build(self, input_shape):
        pass

    def call(self, inputs, training=True):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs

class EnsembleClassifier:
    """ Implementation of an ensemble classifier """
    def __init__(self,
                 modelsList,
                 classNames,
                 imageSize=(224, 224),
                 dropoutPercent=0.2,
                 monteCarloRuns=20,
                 batchSize=16):
        """
        :param modelsList: List of models to use as the ensemble
        :param imageSize: Image size to feed into the network
        :param dropoutPercent: Percentage to use in each network run
        """
        num_models = len(modelsList)
        assert num_models > 1, "Ensemble classifier requires more than 1 model"
        assert classNames > 1, "Must supply more than 1 class name"

        self._input_shape = (*imageSize, 3)
        self._class_names = classNames
        self._num_mc_runs = monteCarloRuns
        self._models = modelsList
        self._batch_size = batchSize

        label_batch = np.zeros((1, len(classNames)))

        # ## Create the model
        fe_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        fe_layer = hub.KerasLayer(fe_url, input_shape=self._input_shape)
        fe_layer.trainable = False

        dropout1 = Dropout_MC(dropoutPercent)
        dropout2 = Dropout_MC(dropoutPercent)
        pred_layer1 = layers.Dense(1280, activation='relu')
        pred_layer2 = layers.Dense(label_batch.shape[1])
        softmax = tf.keras.layers.Softmax()

        self._model_2 = tf.keras.Sequential([
            fe_layer,
            dropout1,
            pred_layer1,
            dropout1,
            pred_layer1,
            dropout2,
            pred_layer2,
            softmax
        ])
        self._model_2.trainable = False

    def resize_images(self, images):
        """ Resize a list of images to the network input size """
        resized = []
        for image in images:
            resized.append(cv2.resize(image, dsize=self._input_shape[:2]))
        return resized
    def run_track(self, track_images):
        """
        Pass in a list of images belonging to a track
        :param images: list of images to process
        """
        num_batches = np.ceil(len(track_images) / self._batch_size)
        ensemble_sum = np.zeros((len(track_images), len(self._class_names)),
                                dtype='float32')
        resized_images = self.resize_images(track_images)
        # For each model, run each for the number of monte-carlo runs
        for model in self._models:
            self._model_2.load_weights(model)
            for _ in range(self._num_mc_runs):
                for idx in range(int(num_batches)):
                    start = self._batch_size*idx
                    end = start + self._batch_size
                    current_batch = resized_images[start:end]
                    current_results = self._model_2.predict_on_batch(current_batch)
                    ensemble_sum[start:end, :] = np.add(ensemble_sum[start:end, :],
                                                        current_results)


        ensemble_score_vec = ensemble_sum / (len(self._models)*self._num_mc_runs)

        calc_entropy = lambda score: -1*np.sum(np.asarray([x * np.log(x) for x in score]))
        entropy_vec = [calc_entropy(score) for score in ensemble_score_vec]
        return ensemble_score_vec, entropy_vec

    def process_track_results(self,
                              ensemble_score_vec,
                              entropy_vec,
                              entropy_cutoff=0.40,
                              high_entropy_name='unknown'):
        """
        Process an series of ensemble scores and entropy values to classify a collection of images
        """

        # Generate winner for each image by the max of the score vector
        labels = np.argmax(ensemble_score_vec, axis=0)
        track_sum = np.zeros(len(self._class_names))
        for label, entropy in zip(labels, entropy_vec):
            track_sum[label] += 1 - float(entropy)

        winner = np.argmax(track_sum)
        if np.max(track_sum) > entropy_cutoff:
            label = high_entropy_name
        else:
            label = self._class_names[winner]

        # return label, but ingredients to if some diagnostics are useful to
        # higher level callers
        return label, winner, track_sum
