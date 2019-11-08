import unittest
import os
from openem.Detect import SSDDetector
import cv2
import numpy as np
import tensorflow as tf

class DetectionTest(tf.test.TestCase):
    def setUp(self):
        self.deploy_dir = os.getenv('deploy_dir')
        if self.deploy_dir == None:
            raise 'Must set ENV:deploy_dir'

        self.ruler_dir = os.path.join(self.deploy_dir, "detect")
        self.pb_file = os.path.join(self.ruler_dir,"detect.pb")

        # Test image definitions and expected values
        self.images=['test_image_000.jpg',
                     'test_image_001.jpg',
                     'test_image_002.jpg']
        self.fishLocations=[
            [144.15286255, 102.43278503, 163.65844727, 104.77685547],
            [],
            []
        ]

    def test_correctness(self):
        finder=SSDDetector(self.pb_file)
        for idx,image in enumerate(self.images):
            image_data=cv2.imread(os.path.join(self.ruler_dir,
                                  image))
            finder.addImage(image_data)

        # Verify the same thing but in batch mode
        batch_result = finder.process()
        self.assertIsNotNone(batch_result)
        self.assertEqual(batch_result.shape[0],len(self.images))
        for idx in range(len(self.images)):
            with self.subTest(idx=idx):
                location=batch_result[idx]
                self.assertAlmostEqual(location, self.fishLocations[idx],
                                       msg=f"Failed image: {location}")
