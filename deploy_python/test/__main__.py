import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from test.FindRulerTest import FindRulerTest
from test.DetectionTest import DetectionTest

if __name__=="__main__":
    tf.test.main()
