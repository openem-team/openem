import unittest
import os
from openem.FindRuler import RulerMaskFinder
import openem.FindRuler
import cv2
import numpy as np
import tensorflow as tf

class FindRulerTest(tf.test.TestCase):
    def setUp(self):
        self.deploy_dir = os.getenv('deploy_dir')
        if self.deploy_dir == None:
            raise 'Must set ENV:deploy_dir'

        self.ruler_dir = os.path.join(self.deploy_dir, "find_ruler")
        self.pb_file = os.path.join(self.ruler_dir,"find_ruler.pb")

        # Test image definitions and expected values
        self.images=['test_image_000.jpg',
                     'test_image_001.jpg',
                     'test_image_002.jpg']
        # C++ code came up with:
        # [282, 140], [460,368]
        # [415, 214], [516,217]
        # [38, 306], [338, 311]
        self.rulerCoordinates=[[[283,142],[458,366]],
                               [[415,214],[515,217]],
                               [[38,306],[336,311]]]

        self.expected_hits = [5663, 2243, 5137]

    def test_stability(self):
        finder=RulerMaskFinder(self.pb_file)

        # Test 1 at a time mode
        for idx,image in enumerate(self.images):
            image_data=cv2.imread(os.path.join(self.ruler_dir,
                                           image))
            finder.addImage(image_data)
            image_result = finder.process()
            self.assertIsNotNone(image_result)

            unique,counts = np.unique(image_result[0], return_counts=True)
            histogram = dict(zip(unique,counts))
            hits = histogram[255]
            # Expect answer within 5%
            self.assertAlmostEqual(hits, self.expected_hits[idx],
                                   msg=f"Failed image {idx}: {histogram}",
                                   delta=hits*.01)
            # Code here is how to write mask image
            cv2.imwrite(f'test_{idx}.jpg', image_result[0])

            # Verify ruler is present
            self.assertTrue(openem.FindRuler.rulerPresent(image_result[0]))
            ruler=openem.FindRuler.rulerEndpoints(image_result[0])
            expected=self.rulerCoordinates[idx]
            for x in [0,1]:
                for y in [0,1]:
                    self.assertAlmostEqual(expected[x][y],
                                           ruler[x][y],
                                           msg=f"{x},{y}, {idx} Fail {ruler}",
                                           delta=5)


    def test_batch(self):
        finder=RulerMaskFinder(self.pb_file)
        for idx,image in enumerate(self.images):
            image_data=cv2.imread(os.path.join(self.ruler_dir,
                                  image))
            finder.addImage(image_data)

        # Verify the same thing but in batch mode
        batch_result = finder.process()
        self.assertIsNotNone(batch_result)
        self.assertEqual(batch_result.shape[0],len(self.images))
        for idx in range(len(self.images)):
            unique,counts = np.unique(batch_result[idx], return_counts=True)
            histogram = dict(zip(unique,counts))
            hits = histogram[255]
            self.assertAlmostEqual(hits, self.expected_hits[idx],
                                   msg=f"Failed image {idx}: {histogram}",
                                   delta=hits*.01)
