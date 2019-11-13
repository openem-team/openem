import unittest
import os
from openem.Count import KeyframeFinder
import cv2
import numpy as np
import tensorflow as tf

class ClassifyTest(tf.test.TestCase):
    def setUp(self):
        self.deploy_dir = os.getenv('deploy_dir')
        if self.deploy_dir == None:
            raise 'Must set ENV:deploy_dir'

        self.ruler_dir = os.path.join(self.deploy_dir, "count")
        self.pb_file = os.path.join(self.ruler_dir,"count.pb")

    def test_correctness(self):
        # Test files have dims of 720,360
        finder=KeyframeFinder(self.pb_file, 720, 360)
        self.assertIsNotNone(finder)
