#!/usr/bin/env python

__copyright__ = "Copyright (C) 2018 CVision AI."
__license__ = "GPLv3"
# This file is part of OpenEM, released under GPLv3.
# OpenEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

"""Top level script for training new OpenEM models.
"""

import argparse
import sys
from openem_train import preprocess
from openem_train import find_ruler
from openem_train import detect
from openem_train import classify
from openem_train import count
from openem_train import test
from openem_train.util.config_interface import ConfigInterface

def main():
    """Parses command line args and trains models.
    """
    # Add the deployment library to path.
    sys.path.append('../python')
    parser = argparse.ArgumentParser(
        description="Top level training script.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'config_file',
        help="Path to config file."
    )
    parser.add_argument(
        'task',
        help="What task to do, one of: \n"
        "extract_images: Convert relevant video frames to images.\n"
        "find_ruler_train: Train algorithm to find ruler.\n"
        "find_ruler_predict: Predict ruler position for extracted images.\n"
        "extract_rois: Use predicted ruler locations to extract ROIs.\n"
        "detect_train: Train algorithm to detect fish.\n"
        "detect_predict: Predict fish locations for extracted ROIs.\n"
        "extract_dets: Use predicted fish locations to extract detections.\n"
        "classify_train: Train algorithm to classify fish.\n"
        "classify_predict: Predict fish species for extracted detections.\n"
        "count_train: Train algorithm to count fish.\n"
        "test: Test a fully trained model.\n"
        "all: Do all of the above in sequence."
    )
    args = parser.parse_args()

    # Read in the config file.
    config = ConfigInterface(args.config_file)

    do_all = args.task == 'all'

    if (args.task == 'extract_images') or do_all:
        preprocess.extract_images(config)

    if (args.task == 'find_ruler_train') or do_all:
        find_ruler.train(config)

    if (args.task == 'find_ruler_predict') or do_all:
        find_ruler.predict(config)

    if (args.task == 'extract_rois') or do_all:
        preprocess.extract_rois(config)

    if (args.task == 'detect_train') or do_all:
        detect.train(config)

    if (args.task == 'detect_predict') or do_all:
        detect.predict(config)

    if (args.task == 'extract_dets') or do_all:
        preprocess.extract_dets(config)

    if (args.task == 'classify_train') or do_all:
        classify.train(config)

    if (args.task == 'classify_predict') or do_all:
        classify.predict(config)

    if (args.task == 'count_train') or do_all:
        count.train(config)

    if (args.task == 'test') or do_all:
        test.predict(config)

if __name__ == '__main__':
    main()
