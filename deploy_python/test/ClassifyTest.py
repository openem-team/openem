import unittest
import os
from openem.Classify import Classifier
import cv2
import numpy as np
import tensorflow as tf

class ClassifyTest(tf.test.TestCase):
    def setUp(self):
        self.deploy_dir = os.getenv('deploy_dir')
        if self.deploy_dir == None:
            raise 'Must set ENV:deploy_dir'

        self.ruler_dir = os.path.join(self.deploy_dir, "classify")
        self.pb_file = os.path.join(self.ruler_dir,"classify.pb")

        # Test image definitions and expected values
        self.images=['test_image_000.jpg',
                     'test_image_001.jpg',
                     'test_image_002.jpg',
                     'test_image_003.jpg',
                     'test_image_004.jpg']
        self.species=[
            #Window pane flounder
            np.array([1.5273033e-08,
                      2.0885507e-07,
                      2.3328525e-07,
                      2.8659571e-07,
                      8.1960671e-08,
                      2.9835022e-05,
                      9.9996936e-01,
                      2.2464786e-08]),
            #Window pane flounder
            np.array([8.2368363e-04,
                      1.8096182e-05,
                      7.4708383e-05,
                      2.4042622e-05,
                      4.2192089e-05,
                      1.1589836e-04,
                      9.9861109e-01,
                      2.9028734e-04]),
            #Other
            np.array([1.1799466e-08,
                      9.7716629e-08,
                      9.9992871e-01,
                      5.0134649e-06,
                      1.8012923e-05,
                      5.8652012e-07,
                      4.7377016e-05,
                      1.7026829e-07]),
            #Summer flounder
            np.array([1.1247632e-04,
                      1.7128131e-06,
                      5.3557684e-05,
                      5.7094945e-05,
                      9.9974149e-01,
                      6.8521052e-07,
                      2.6996197e-05,
                      5.8836113e-06]),
            #Summer flounder
            np.array([1.10131885e-04,
                       6.85039122e-05,
                       1.60592899e-05,
                       2.36838032e-03,
                       9.97359455e-01,
                       4.89681588e-06,
                       3.45146909e-05,
                       3.79584453e-05])]
        
        # Clear, Cover, Clear, Cover, Clear
        self.covers=[np.array([4.0111686e-08, 4.7601839e-06, 9.9999523e-01]),
                     np.array([0.00119321, 0.9960699 , 0.00273687]),
                     np.array([1.1223665e-07, 5.4563090e-05, 9.9994528e-01]),
                     np.array([0.00626855, 0.9531476 , 0.04058384]),
                     np.array([4.0183782e-05, 2.6267747e-04, 9.9969721e-01])]

    def test_correctness(self):
        finder=Classifier(self.pb_file, batch_size=len(self.images))
        for idx,image in enumerate(self.images):
            image_data=cv2.imread(os.path.join(self.ruler_dir,
                                  image))
            finder.addImage(image_data)

        # Verify the same thing but in batch mode
        batch_result = finder.process()
        # Verify we got 2 tensor outputs
        self.assertIsNotNone(batch_result)
        num_images = len(self.images)
        self.assertEqual(len(batch_result),5)
        
        for idx in range(num_images):
            with self.subTest(idx=idx):
                classification=batch_result[idx]
                self.assertAllClose(self.species[idx],
                                    classification.species,
                                    rtol=0.10)
                self.assertAllClose(self.covers[idx],
                                    classification.cover,
                                    rtol=0.10)
                
