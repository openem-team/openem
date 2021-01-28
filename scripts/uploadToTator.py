#!/usr/bin/env python3

""" Upload a result set or training set to tator for analysis """
import argparse
import tator
import csv
import progressbar
import sys
import signal
import os
import configparser
import pandas as pd
import datetime
import time
import traceback
import math

from functools import partial

def exit_func(_,__):
    print("SIGINT detected")
    os._exit(0)

detect_count = 0
def process_box(row,
                args,
                tator,
                version_id,
                species_names,
                truth_data,
                bar,
                default_obj_fields):
    global detect_count
    detect_count += 1
    bar.update(detect_count)
    media_element = row['media_element']
    if media_element == None:
        print("ERROR: Could not find media element!")

    obj = None
    if media_element and args.localization_type_id:
        species_id_0 = int(float(row['species_id'])-1)
        add=True
        if args.threshold:
            if confidence < args.threshold:
                add=False
        if add:
            obj = make_localization_obj(args,
                                   tator,
                                   version_id,
                                   args.localization_type_id,
                                   media_element,
                                   default_obj_fields,
                                   frame=int(row['frame']),
                                   x=float(row['x']),
                                   y=float(row['y']),
                                   width=float(row['width']),
                                   height=float(row['height']),
                                   confidence=None,
                                   species=species_names[species_id_0])
    return obj

def process_line(args, tator, species_names, row):
    print("ERROR: Line mode --- Not supported")

def process_detect(row,
                   args,
                   api,
                   version_id,
                   species_names,
                   truth_data,
                   bar,
                   default_obj_fields):
    global detect_count
    detect_count += 1
    bar.update(detect_count)
    if type(truth_data) != type(None):
        if row['frame'] == '':
            return
        match=truth_data.loc[(truth_data.video_id == row['video_id']) & (truth_data.frame == int(row['frame']))]
        if len(match) == 0:
            return
        else:
            pass

    media_element = row['media_element']

    if media_element == None:
        print("ERROR: Could not find media element!")

    obj = None
    if media_element and args.localization_type_id:
        species_id_0 = int(float(row['det_species'])-1)
        if type(row['det_conf']) is str:
            confidence = float(row['det_conf'].split(':')[species_id_0])
        else:
            confidence = row['det_conf']
        add=True
        if args.threshold:
            if confidence < args.threshold:
                add=False
        if add:
            obj = make_localization_obj(args,
                                   tator,
                                   version_id,
                                   args.localization_type_id,
                                   media_element,
                                   default_obj_fields,
                                   frame=int(row['frame']),
                                   x=float(row['x']),
                                   y=float(row['y']),
                                   width=float(row['w']),
                                   height=float(row['h']),
                                   confidence=confidence,
                                   species=species_names[species_id_0])
    return obj

def make_localization_obj(args,
                          tator,
                          version_id,
                          box_type,
                          media_el,
                          default_obj_fields,
                          frame,
                          x,y,width,height,
                          confidence,
                          species):
    obj={"type": box_type,
         "media_id": media_el['id'],
         "x" : x / media_el['width'],
         "y": y / media_el['height'],
         "width": min(1.0,width / media_el['width']),
         "height": min(1.0,height / media_el['height'])
         }

    if version_id:
        obj.update({"version": version_id})


    #default_schema = tator.LocalizationType.byTypeId(box_type)
    #default_schema_columns = [(elem.get('name'),elem.get('default')) for elem in default_schema.get('columns')]
    for col in default_obj_fields:
        obj.update({col[0] : col[1]})

    obj.update({args.species_attr_name : species})

    if confidence:
        obj.update({"Confidence": confidence})
    if args.media_type != "image":
        obj.update({"frame": frame})
    return obj


media_list_cache={}
media_count = 0
def uploadMedia(row, args, api, bar):
    """ Attempts to upload the media in the row to tator
        if already there, skips, but either way returns the
        media element information. """
    video_id = row['video_id']
    frame = row['frame']
    vid_dir=os.path.join(args.img_base_dir, video_id)

    global media_list_cache
    global media_count

    media_count += 1
    bar.update(media_count)

    try:
        img_file=f"{int(frame):04d}.{args.img_ext}"
    except:
        print(f"Skipping {row}")
        return None,None
    img_path=os.path.join(vid_dir, img_file)

    if args.media_type == "pipeline":
        media_id = int(video_id.split('_')[0])
        if media_id in media_list_cache:
            return media_id,media_list_cache[media_id]
        else:
            result = api.get_media(media_id).to_dict()
            media_list_cache[media_id] = result
            return result['id'],result
    elif args.media_type == "image":
        desired_name = f"{row['video_id']}.{args.img_ext}"
    else:
        desired_name = f"{row['video_id']}.{args.img_ext}"

    if desired_name in media_list_cache:
        cache_result = media_list_cache[desired_name]
        return cache_result['id'], cache_result
    else:
        print(f"{time.time()}: {desired_name}: Not In Cache")
        print(f"Cache = {media_list_cache}")
        media_element_search=api.get_media_list(args.project,name=desired_name)
        if media_element_search == None:
            print(f"Uploading file...{desired_name}")
            for _,__ in tator.util.upload_media(api,
                                                args.media_type_id,
                                                img_path,
                                                section=args.section,
                                                fname=desired_name):
                pass
            
            media_element_search=api.get_media_list(args.project,name=desired_name)
            while media_element_search is None:
                time.sleep(1)
                print("Waiting for upload")
                media_element_search=api.get_media_list(args.project,name=desired_name)
        if media_element_search:
            result = media_element_search[0]
            media_list_cache[desired_name] = result
            return result['id'], result
        else:
            return None,None



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = tator.get_parser(parser)
    parser.add_argument("csvfile", help="test.csv, length.csv, or detect.csv")
    parser.add_argument("--project", help="Project ID", type=int)
    parser.add_argument("--img-base-dir", help="Base Path to media files", required=True)
    parser.add_argument("--img-ext", default="jpg")
    parser.add_argument("--media-type-id", type=int, required=True)
    parser.add_argument("--media-type",
                        type=str,
                        choices=["pipeline", "image","video"],
                        default="image")
    parser.add_argument("--species-attr-name", type=str, default="Species")
    parser.add_argument("--confidence-attr-name", type=str, default="Confidence")
    parser.add_argument("--localization-type-id", type=int)
    parser.add_argument("--section", help="Section name to apply")
    parser.add_argument("--train-ini", help="If uploading boxes, this is required to convert species id to a string")
    parser.add_argument("--threshold", type=float, help="Discard boxes less than this value")
    parser.add_argument("--truth-data", type=str, help="Path to annotations.csv to exclude non-truth data")
    parser.add_argument("--species-keyname", type=str,default="Species")

    parser.add_argument("--version-id", type=int)
    args = parser.parse_args()
    api = tator.get_api(args.host, args.token)

    # convert supplied version number to version id
    version_id = args.version_id

    signal.signal(signal.SIGINT, exit_func)

    input_data = pd.read_csv(args.csvfile)
    if len(input_data) == 0:
        print("No input data to process")
        sys.exit(0)
    #For testing big sets try this:
    #input_data = input_data.head(500)
    keys = list(input_data.columns)
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

    print(f"Processing {len(input_data)} elements")
    bar = progressbar.ProgressBar(max_value=len(input_data))
                                  #redirect_stdout=True)

    print("Ingesting media...")
    media_info = input_data.apply(uploadMedia,
                                  axis=1,
                                  result_type='expand',
                                  args=(args, api, bar),
                                  raw=False)
    bar.finish()
    input_data = input_data.assign(media_id = media_info[0])
    input_data = input_data.assign(media_element = media_info[1])

    default_obj_fields=[] # not requried anymore
    print("Generating localizations...")

    unique_media = input_data['media_id'].unique()
    print(unique_media)
    bar = progressbar.ProgressBar(redirect_stdout=True,
                                  max_value=len(unique_media))
    for media_id in bar(unique_media):
        local_list=[]
        media_data = input_data.loc[input_data['media_id'] == media_id]
        bar2 = progressbar.ProgressBar(redirect_stdout=True, max_value=len(media_data))
        detect_count = 0
        localizations = media_data.apply(function_map[mode],
                                         args=(args,
                                               tator,
                                               version_id,
                                               species_names,
                                               truth_data,
                                               bar2,
                                               default_obj_fields),
                                         raw=False,
                                         axis=1)
        bar2.finish()
        raw_objects = localizations.values.tolist()
        for response in tator.util.chunked_create(api.create_localization_list,
                                                  args.project,
                                                  localization_spec=raw_objects):
                pass

        try:
            # When complete for a given media update the sentinel value
            api.update_media(media_id, attributes={"Object Detector Processed": str(datetime.datetime.now())})
        except:
            print("Unable to set sentinel attribute")
