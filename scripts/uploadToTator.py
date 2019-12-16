#!/usr/bin/env python3

import argparse
import pytator
import csv
import progressbar
import sys
import signal
import os
import configparser
import pandas as pd

from multiprocessing import Pool, Value
from functools import partial

progress = Value('i', 0)

def exit_func(_,__):
    print("SIGINT detected")
    os._exit(0)

def process_box(args, tator, species_names, row):
    media_element = uploadMedia(args, tator, row)
    if media_element and args.localization_type_id:
        pass
    with progress.get_lock():
        progress.value += 1

def process_line(args, tator, species_names, row):
    print("ERROR: Line mode --- Not supported")

def process_detect(args, tator, species_names, truth_data, row):
    if type(truth_data) != type(None):
        if row['frame'] == '':
            return
        match=truth_data.loc[(truth_data.video_id == row['video_id']) & (truth_data.frame == int(row['frame']))]
        if len(match) == 0:
            return
        else:
            pass
    media_element = uploadMedia(args, tator, row)
    if media_element == None:
        print("ERROR: Could not find media element!")

    if media_element and args.localization_type_id:
        existing_locals = tator.Localization.filter({"media_id": media_element['id'],
                                                     "type": args.localization_type_id})
        species_id_0 = int(float(row['det_species'])-1)

        confidence = float(row['det_conf'])
        add=True
        if args.threshold:
            if confidence < args.threshold:
                add=False
                print(f"Skipping detection {row}")
        if add:
            add_localization(tator,
                         args.localization_type_id,
                         media_element,
                         x=float(row['x']),
                         y=float(row['y']),
                         width=float(row['w']),
                         height=float(row['h']),
                         confidence=confidence,
                         species=species_names[species_id_0])
    with progress.get_lock():
        progress.value += 1

def add_localization(tator,
                     box_type,
                     media_el,
                     x,y,width,height,
                     confidence,
                     species):
    obj={"type": box_type,
         "media_id": media_el['id'],
         "x" : x / media_el['width'],
         "y": y / media_el['height'],
         "width": width / media_el['width'],
         "height": height / media_el['height'],
         "Species": species,
         "Confidence": confidence}
    result = tator.Localization.new(obj)
    print(result)
def uploadMedia(args, tator, row):
    """ Attempts to upload the media in the row to tator
        if already there, skips, but either way returns the
        media element information. """
    vid_dir=os.path.join(args.img_base_dir, row['video_id'])
    try:
        img_file=f"{int(row['frame']):04d}.{args.img_ext}"
    except:
        print(f"Skipping {row}")
        return
    img_path=os.path.join(vid_dir, img_file)
    if not os.path.exists(img_path):
        print(f"Could not find {img_path}")
        return None
    md5 = pytator.md5sum.md5_sum(img_path)
    desired_name=f"{row['video_id']}_{row['frame']}.{args.img_ext}"
    media_element_search=tator.Media.filter({"name": desired_name})
    if media_element_search == None:
        print(f"Uploading file...{desired_name}")
        tator.Media.uploadFile(args.media_type_id,
                               img_path,
                               md5=md5,
                               progressBars=False,
                               section=args.section,
                               fname=desired_name)
        media_element_search=tator.Media.filter({"name": desired_name})
    if media_element_search:
        return tator.Media.get(media_element_search[0]['id'])
    else:
        return None



if __name__=="__main__":
    parser = argparse.ArgumentParser('openem_downloader')
    parser = pytator.tator.cli_parser(parser)
    parser.add_argument("csvfile", help="test.csv, length.csv, or detect.csv")
    parser.add_argument("--img-base-dir", help="Base Path to media files", required=True)
    parser.add_argument("--img-ext", default="jpg")
    parser.add_argument("--media-type-id", type=int, required=True)
    parser.add_argument("--localization-type-id", type=int)
    parser.add_argument("--section", help="Section name to apply")
    parser.add_argument("--pool-size", type=int, default=4, help="Number of threads to use")
    parser.add_argument("--train-ini", help="If uploading boxes, this is required to convert species id to a string")
    parser.add_argument("--threshold", type=float, help="Discard boxes less than this value")
    parser.add_argument("--truth-data", type=str, help="Path to annotations.csv to exclude non-truth data")
    args = parser.parse_args()
    tator = pytator.Tator(args.url, args.token, args.project)

    signal.signal(signal.SIGINT, exit_func)

    csv_file = open(args.csvfile, 'r')
    input_data_reader = csv.DictReader(csv_file)
    keys = input_data_reader.fieldnames
    boxes_keys = ['video_id','frame','x','y','width','height','theta','species_id']
    lines_keys = ['video_id','frame','x1','y1','x2','y2','species_id']
    detect_keys = ['video_id', 'frame', 'x','y','w','h','det_conf','det_species']

    mode = None
    # Figure out which type of csv file we are dealing with
    if all(x in keys for x in boxes_keys):
        mode = 'box'
    elif all(x in keys for x in lines_keys):
        mode = 'line'
    elif all(x in keys for x in detect_keys):
        mode = 'detect'
    else:
        print(f"ERROR: Can't deduce file type from {keys}")
        sys.exit(-1)

    # Function map
    function_map={'box': process_box,
                  'line': process_line,
                  'detect': process_detect}

    species_names = None
    if args.train_ini:
        config = configparser.ConfigParser()
        config.read(args.train_ini)
        species_names=config.get('Data', 'Species').split(',')

    truth_data = None
    if args.truth_data:
        truth_data = pd.read_csv(args.truth_data)
    partial_func = partial(function_map[mode], args, tator,
                           species_names, truth_data)
    pool=Pool(processes=args.pool_size)
    input_data = list(input_data_reader)
    print(f"Processing {len(input_data)} elements")
    #result = pool.map_async(partial_func, input_data)
    bar = progressbar.ProgressBar(max_value=len(input_data),
                                  redirect_stdout=True)
    for row in bar(input_data):
        partial_func(row)
    #while result.ready() == False:
    #    bar.update(progress.value)
    #    result.wait(1)
