#!/usr/bin/env python3

import sys
import unittest
sys.path.append("../train")
from collections import namedtuple
import math

def areaOfBox(box):
    s1=math.sqrt(math.pow(box[0][0]-box[1][0],2)+
                 math.pow(box[0][1]-box[1][1],2))
    s2=math.sqrt(math.pow(box[1][0]-box[2][0],2)+
                 math.pow(box[1][1]-box[2][1],2))
    return s1*s2
    
class TestUtils (unittest.TestCase):
    def testTransform(self):
        from openem_train.util.utils import rotate_detection
        # avoid import of tensorflow and what not
        FishBoxDetection = namedtuple(
            'FishBoxDetection',
            ['x', 'y', 'width', 'height', 'theta'])

        # Testing rotation logic:
        # Generate a detection at various points of various sizes
        # rotate it and verify the area is always the same
        for x in range(10,300,50):
            for y in range(10,300,50):
                for width in range(10,300,50):
                    for height in range(10,300,50):
                        trueArea=width*height
                        for phi in range(0,400, 7):
                            detection=FishBoxDetection(x=x,
                                                       y=y,
                                                       width=width,
                                                       height=height,
                                                       theta=phi)
                            areaOfRotation=areaOfBox(rotate_detection(detection))
                            self.assertTrue(math.isclose(areaOfRotation, trueArea))
        
        

if __name__=="__main__":
    unittest.main()
