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
            self.assertAllClose(expected,
                                ruler,
                                msg=f"{idx} Fail: {ruler}",
                                atol=5)


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

    def test_errorHandling(self):
        finder=RulerMaskFinder(self.pb_file)
        self.assertIsNone(finder.process())

class RoiTests(tf.test.TestCase):
    """ Tests that don't actually use the tensorflow model
        but their nice *All* functions are convenient for any
        vector-based assertion
    """
    def test_RoiLogic(self):
        # Define an 8x8 bitmap with a 2x2 box in the middle
        img=np.zeros((8,8))
        img[2:6,2:6]=np.ones((4,4))

        expected=[2,2,4,4]
        bb_roi = openem.FindRuler.findRoi(img,0)
        self.assertAllEqual(expected, bb_roi)

        expected=[1,1,6,6]
        bb_roi = openem.FindRuler.findRoi(img,1)
        self.assertAllEqual(expected, bb_roi)

        expected=[0,0,8,8]
        bb_roi = openem.FindRuler.findRoi(img,2)
        self.assertAllEqual(expected, bb_roi)

        # Error case here; if margin exceeds image
        expected=[0,0,8,8]
        bb_roi = openem.FindRuler.findRoi(img,3)
        self.assertAllEqual(expected, bb_roi, msg="Margin = 3")

    def test_crop(self):
        # Define an 8x8 bitmap with a 2x2 box in the middle
        img=np.zeros((8,8))
        img[2:6,2:6]=np.ones((4,4))

        bb_roi = openem.FindRuler.findRoi(img,0)
        crop=openem.FindRuler.crop(img, bb_roi)
        self.assertAllEqual(crop, np.ones((4,4)))
