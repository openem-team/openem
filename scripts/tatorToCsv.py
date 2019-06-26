#!/usr/bin/env python3

import argparse
import json
import pandas as pd
import os
import os.path
from collections import namedtuple
import progressbar
import configparser
import numpy as np
import matplotlib.pyplot as plt

FishBoxDetection = namedtuple(
    'FishBoxDetection',
    ['video_id', 'frame', 'x', 'y', 'width', 'height', 'theta', 'species_id'])

def _getScientificName(jsonSpecies):
    scientificName=jsonSpecies.split('(')[0].strip().capitalize()
    return scientificName
def _convertLocalizationsFromFile(inputPath, speciesNames):
    base=os.path.basename(inputPath)
    video_id=os.path.splitext(base)[0]

    # List of detections in openem tuple format
    oemDetections=[]
    with open(inputPath, 'r') as data:
        obj=json.load(data)
        detections=obj["detections"]
        boxes=0
        ignored=0
        unknown=0
        unknownNames=set()
        
        for detection in detections:
            if detection["type"] == "box":
                name=_getScientificName(detection["species"])
                if name not in speciesNameMap:
                    unknown=unknown+1
                    unknownNames.add(name)
                    continue
                
                oemDetections.append(
                    FishBoxDetection(
                        video_id=video_id,
                        frame=int(detection["frame"]),
                        x=float(detection["x"]),
                        y=float(detection["y"]),
                        width=float(detection["w"]),
                        height=float(detection["h"]),
                        species_id=speciesNameMap[name],
                        theta=0
                    )
                )
                boxes=boxes+1
            else:
                ignored=ignored+1
        
    return (oemDetections, (boxes,ignored, unknown, unknownNames))

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Converts tator json annotations to csv")  
    parser.add_argument("-i", "--input",
                        help="Path to input file")
    parser.add_argument("-o", "--output",
                        help="Path to input file",
                        required=True)
    parser.add_argument("-t", "--testOutput",
                        help="Path to input file",
                        required=True)
    parser.add_argument("-c", "--config",
                        help="Path to openem train.ini",
                        required=True)
    parser.add_argument("-d", "--directory",
                        help="Path to input file")

    args=parser.parse_args()

    if args.directory and args.input:
        print("ERROR: Can't supply both directory(--directory) and file(--input) inputs");
        parser.print_help()
        sys.exit(-1)
    if args.directory == None and args.input == None:
        print("ERROR: Must supply either directory(--directory) of file(--input) inputs");
        parser.print_help()
        sys.exit(-1)

    config=configparser.ConfigParser()
    config.read(args.config)
    speciesNames=config.get("Data", "Species").split(",")
    speciesNameMap={}
    for idx,name in enumerate(speciesNames):
        speciesNameMap[name] = idx+1

    data=[]
    if args.input:
        fileData, stats=_convertLocalizationsFromFile(args.input,
                                                      speciesNameMap)
        print(f"Processed {stats[0]} box localizations")
        print(f"Ignored {stats[1]} localizations due to wrong type")
        if stats[2]:
            print(f"Ignored {stats[2]} localizations due to unknown species")
            print(f"Unknown names = {stats[3]}")
        data.extend(fileData)
    else:
        dirContents=os.listdir(args.directory)
        filesToProcess=[]

        for fname in dirContents:
            comps=os.path.splitext(fname)
            if len(comps) > 1:
                if comps[1][1:] == 'json':
                    filesToProcess.append(fname)
        progressbar.streams.wrap_stderr()
        bar=progressbar.ProgressBar(prefix='Files',
                                    redirect_stdout=True,
                                    redirect_stderr=True)
        stats=np.zeros(3)
        unknownNames=set()
        for fname in bar(filesToProcess):
            fileData, fileStats=_convertLocalizationsFromFile(
                os.path.join(args.directory,fname),
                speciesNameMap)
            data.extend(fileData)
            stats=stats+np.array(fileStats[:3])
            unknownNames=unknownNames.union(fileStats[3])
        print(f"Processed {stats[0]} box localizations")
        print(f"Ignored {stats[1]} localizations due to wrong type (i.e. dot)")
        if stats[2]:
            print(f"Ignored {stats[2]} localizations due to unknown species")
            print(f"Unknown names = {unknownNames}")


    df=pd.DataFrame(columns=FishBoxDetection._fields,
                 data=data)
    train=df.sample(frac=0.9, random_state=200)
    train.to_csv(args.output,index=False)
    test=df.drop(train.index)
    test.to_csv(args.testOutput, index=False)
