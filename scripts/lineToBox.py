#!/usr/bin/env python3

import argparse
import configparser
import math
import numpy as np
import pandas as pd
from collections import namedtuple
import sys

FishBoxDetection = namedtuple(
    'FishBoxDetection',
    ['video_id', 'frame', 'x', 'y', 'width', 'height', 'theta', 'species_id'])

def processTransformation(inputFile, outputFile, aspectRatios):
    inputFrame=pd.read_csv(inputFile)
    frameBuffer=[]
    for _,row in inputFrame.iterrows():
        if row.species_id != 0:
            width=math.hypot(row.x2-row.x1,row.y2-row.y1)
            # Aspect ratio is height / width despite comment in ini file!
            aspectRatio=float(aspectRatios[row.species_id])
            height=width * aspectRatio * 2

            # Calculate theta of the box atan (in radians)
            theta=math.atan2(row.y2-row.y1,row.x2-row.x1)
            # phi is the angle of the perpendicular line
            # It is also the angle of shift to apply to x1,y1 to get x,y

            # Construct and utilize a translation matrix
            # Start can flip to keep the box right-handed
            start=[row.x1,row.y1]

            # If we are pitched higher than 90 degrees normalize to make
            # box logic easier and swap start/finish to invert direction
            if theta > (math.pi / 2):
                theta=theta-math.pi
                start=[row.x2,row.y2]
            elif theta < -(math.pi/2):
                theta=theta+math.pi
                start=[row.x2,row.y2]
            elif math.isclose(theta, math.pi/2):
                pass
            elif math.isclose(theta, -math.pi/2):
                start=[row.x2,row.y2]
                theta=math.pi/2

            phi=theta-(math.pi/2)
            xShift=math.cos(phi)*(height/2)
            yShift=math.sin(phi)*(height/2)
            Translation=np.array([[1,0,xShift],
                                  [0,1,yShift],
                                  [0,0,1]])
            xyz=Translation.dot([start[0],start[1],1])

            frameBuffer.append(FishBoxDetection(video_id=row.video_id,
                                                frame=row.frame,
                                                x=xyz[0],
                                                y=xyz[1],
                                                width=width,
                                                height=height,
                                                theta=theta,
                                                species_id=row.species_id))
        else:
            frameBuffer.append(FishBoxDetection(video_id=row.video_id,
                                                frame=row.frame,
                                                x=None,
                                                y=None,
                                                width=None,
                                                height=None,
                                                theta=None,
                                                species_id=row.species_id))

    outputFrame=pd.DataFrame(data=frameBuffer,
                             columns=list(FishBoxDetection._fields))
    outputFrame.to_csv(outputFile, index=False)

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

    config=configparser.ConfigParser()
    config.read(args.config)
    ratios=config.get('Data', 'AspectRatios').split(',')
    outputFile=sys.stdout
    needToClose=False
    if args.output:
        outputFile=open(args.output, 'w')
        needToClose=True

    with open(args.input, 'r') as inputFile:
        processTransformation(inputFile, outputFile, ratios)
    if needToClose:
        outputFile.close()
