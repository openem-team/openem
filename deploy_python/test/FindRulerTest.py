import unittest
import os
from openem.FindRuler import RulerMaskFinder
import cv2
import numpy as np

class FindRulerTest(unittest.TestCase):
    def setUp(self):
        self.deploy_dir = os.getenv('deploy_dir')
        if self.deploy_dir == None:
            raise 'Must set ENV:deploy_dir'

        self.ruler_dir = os.path.join(self.deploy_dir, "find_ruler")
        
    def test_topLevel(self):
        pb_file = os.path.join(self.ruler_dir,"find_ruler.pb")
        finder=RulerMaskFinder(pb_file)

        images=['test_image_000.jpg',
                'test_image_001.jpg',
                'test_image_002.jpg']

        
        # Test 1 at a time mode
        for idx,image in enumerate(images):
            image_data=cv2.imread(os.path.join(self.ruler_dir,
                                           image))
            finder.addImage(image_data)
            image_result = finder.process()
            self.assertIsNotNone(image_result)
            print(f"Shape = {image_result[0].shape}")

            # Make debug image real quick
            cv2.imwrite(f"test_{idx}.jpg", image_result[0])
            
                                  
