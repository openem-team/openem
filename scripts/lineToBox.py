#!/usr/bin/env python3

import argparse
import pandas as pd
from collections import namedtuple
import sys

FishDetection = namedtuple(
    'FishDetection',
    ['video_id', 'frame', 'x1', 'y1', 'x2', 'y2', 'class_id'])
FishBoxDetection = namedtuple(
    'FishBoxDetection',
    ['video_id', 'frame', 'x', 'y', 'width', 'height', 'theta', 'class_id'])

def processTransformation(inputFile, outputFile):
    pass

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("-c", "--config",
                        required=True,
                        help="Path to config file")
    parser.add_argument("-o", "--output",
                        help="Name of output file, default is stdout")
    parser.add_argument("input",
                        help="Name of input file")
    args=parser.parse_args()

    outputFile=sys.stdout
    needToClose=False
    if args.output:
        outputFile=open(args.output, 'w')
        needToClose=True

    with open(args.input, 'r') as inputFile:
        processTransformation(inputFile, outputFile)
    if needToClose:
        outputFile.close()
