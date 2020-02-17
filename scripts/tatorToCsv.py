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

Cover=namedtuple(
    'Cover',
    ['video_id', 'frame', 'cover'])

def _getScientificName(jsonSpecies):
    scientificName=jsonSpecies.split('(')[0].strip().capitalize()
    return scientificName

def _getTrackQuality(tracks, frame : int):
    """
    ignore=hand covering
    entering=clear (or not found)
    exiting=no view
    """
    trackAtFrame=None
    quality=2
    for track in tracks:
       if int(track["frame_added"]) == frame:
           trackAtFrame=trackoemCoversoemCovers
           break
    if trackAtFrame:
        if trackAtFrame["count_label"] == "entering":
            quality=2
        elif trackAtFrame["count_label"] == "ignore":
            quality=1
        else:
            quality=0

    return quality

def check_file(detections):
    for detection in detections:
        if detection["type"] == "box" or detection["type"] == "line":
            return True
    return False

def convert_line_ann_to_box(ann):
    """Convert a line annotation to a bounding box one
    """
    x1 = min(int(ann['x']),int(ann['w']))
    x2 = max(int(ann['x']),int(ann['w']))
    y1 = min(int(ann['y']),int(ann['h']))
    y2 = max(int(ann['y']),int(ann['h']))
    assert(x2>=x1)
    assert(y2>=y1)
    z = 1.1 * np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    nx = x1 + (x2 - x1)/2 - z/2
    ny = y1 + (y2 - y1)/2 - z/2
    new_ann = ann
    new_ann['image_file'] = ann['image_file']
    new_ann['type'] = 'box'
    new_ann['x'] = str(int(round(nx)))
    new_ann['y'] = str(int(round(ny)))
    new_ann['w'] = str(int(round(z)))
    new_ann['h'] = str(int(round(z)))
    return new_ann

def generate_box(detection, speciesNameMap, video_id, createCover, oemCovers, oemDetections):
    try:
        frame=int(detection["frame"])
    except:
        frame=0
    oemDetections.append(
        FishBoxDetection(
            video_id=video_id,
            frame=frame,
            x=float(detection["x"]),
            y=float(detection["y"]),
            width=float(detection["w"]),
            height=float(detection["h"]),
            species_id=speciesNameMap[name],
            theta=0
        )
    )

    #For now assume all localizations are non-covered
    if createCover:
        oemCovers.append(
            Cover(video_id=video_id,
                frame=frame,
                cover=_getTrackQuality(tracks,frame)
            )
        )

    return oemDetections, oemCovers

def check_species(detection, unknown, unknownNames, speciesNameMap):
    name=_getScientificName(detection["species"])
    print(name)
    if name not in speciesNameMap:
        unknown+=1
        unknownNames.add(name)
        return True, unknown, unknownNames
    return False, unknown, unknownNames

def _convertLocalizationsFromFile(inputPath, speciesNames, createCover):
    base=os.path.basename(inputPath)
    video_id=os.path.splitext(base)[0]

    # List of detections in openem tuple format
    oemDetections=[]
    oemCovers=[]
    with open(inputPath, 'r') as data:
        obj=json.load(data)
        detections=obj["detections"]
        
        if createCover:
            tracks=obj["tracks"]
        boxes=0
        lines=0
        ignored=0
        unknown=0
        unknownNames=set()

        if check_file(detections):
            for detection in detections:
                if detection["type"] == "box":
                    species, unknown, unknownNames = check_species(detection, unknown, unknownNames, speciesNameMap)
                    if species:
                        continue
                    oemDetections, oemCovers = generate_box(detection, speciesNameMap, video_id, createCover, oemCovers, oemDetections)
                    boxes +=1
                elif detection["type"] == "line":
                    species, unknown, unknownNames = check_species(detection, unknown, unknownNames, speciesNameMap)
                    if species:
                        continue
                    #Generate box notations
                    detection=convert_line_ann_to_box(detection)
                    oemDetections, oemCovers = generate_box(detection, speciesNameMap, video_id, createCover, oemCovers, oemDetections)
                    lines += 1
        else:
            ignored+=1

    return (oemDetections, (boxes,ignored, unknown, unknownNames, lines), oemCovers)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Converts tator json annotations to csv.")
    parser.add_argument("-i", "--input",
                        help="Path to input file.  If you wish to input directory, please use --directory.")
    parser.add_argument("--lengthOutput",
                        help="Path to output length file.",
                        required=True)
    parser.add_argument("--coverOutput",
                        help="Path to output cover file.")
    parser.add_argument("-t", "--testOutput",
                        help="Path to test file.",
                        required=True)
    parser.add_argument("-c", "--config",
                        help="Path to openem train.ini.",
                        required=True)
    parser.add_argument("-d", "--directory",
                        help="Path to directory.  If you wish to use and input file, please use input.")

    args=parser.parse_args()

    if args.directory and args.input:
        print("ERROR: Can't supply both directory(--directory) and file(--input) inputs");
        parser.print_help()
        sys.exit(-1)
    if args.directory == None and args.input == None:
        print("ERROR: Must supply either directory(--directory) of file(--input) inputs");
        parser.print_help()
        sys.exit(-1)

    createCover = True if args.coverOutput else False

    config=configparser.ConfigParser()
    config.read(args.config)
    speciesNames=config.get("Data", "Species").split(",")
    speciesNameMap={}
    for idx,name in enumerate(speciesNames):
        speciesNameMap[name] = idx+1

    data=[]
    covers=[]
    if args.input:
        fileData, stats=_convertLocalizationsFromFile(args.input,
                                                      speciesNameMap,
                                                      createCover)
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
            fileData, fileStats, cover=_convertLocalizationsFromFile(
                os.path.join(args.directory,fname),
                speciesNameMap, createCover)
            data.extend(fileData)
            covers.extend(cover)

            stats=stats+np.array(fileStats[:3])
            unknownNames=unknownNames.union(fileStats[3])
        print(f"Processed {stats[0]} box localizations")
        print(f"Ignored {stats[1]} localizations due to wrong type (i.e. dot)")
        if stats[2]:
            print(f"Ignored {stats[2]} localizations due to unknown species")
            print(f"Unknown names = {unknownNames}")


    df=pd.DataFrame(columns=FishBoxDetection._fields,
                 data=data)
    videos_list=df['video_id'].unique()
    videos_df=pd.DataFrame(columns=['video_id'],
                            data=videos_list)
    # Sample 90% of the videos
    train_vids=videos_df.sample(frac=0.89, random_state=202)
    train=df.loc[df['video_id'].isin(train_vids["video_id"].tolist())]
    train.to_csv(args.lengthOutput,index=False)
    test=df.drop(train.index)
    test.to_csv(args.testOutput, index=False)

    if createCover:
        coverDf=pd.DataFrame(columns=Cover._fields,
                            data=covers)
        coverDf.to_csv(args.coverOutput,index=False)
