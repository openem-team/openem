#!/usr/bin/env python3

""" Upload a result set or training set to tator for analysis """
import argparse
import pytator
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

from functools import partial

def exit_func(_,__):
    print("SIGINT detected")
    os._exit(0)

def process_box(args, tator, species_names, truth_data, media_map, row):
    media_element = media_map[row['media_id']]
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
                                   args.localization_type_id,
                                   media_element,
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

def process_detect(args, tator, species_names, truth_data, media_map, row):
    if type(truth_data) != type(None):
        if row['frame'] == '':
            return
        match=truth_data.loc[(truth_data.video_id == row['video_id']) & (truth_data.frame == int(row['frame']))]
        if len(match) == 0:
            return
        else:
            pass
    media_element = media_map[row['media_id']]
    if media_element == None:
        print("ERROR: Could not find media element!")

    obj = None
    if media_element and args.localization_type_id:
        species_id_0 = int(float(row['det_species'])-1)
        confidence = float(row['det_conf'])
        add=True
        if args.threshold:
            if confidence < args.threshold:
                add=False
        if add:
            obj = make_localization_obj(args,
                                   tator,
                                   args.localization_type_id,
                                   media_element,
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
                          box_type,
                          media_el,
                          frame,
                          x,y,width,height,
                          confidence,
                          species):
    obj={"type": box_type,
         "media_id": media_el['id'],
         "x" : x / media_el['width'],
         "y": y / media_el['height'],
         "width": width / media_el['width'],
         "height": height / media_el['height'],
         args.species_keyname: species}
    if confidence:
        obj.update({"Confidence": confidence})
    if args.media_type != "image":
        obj.update({"frame": frame})
    return obj


media_list_cache={}
def uploadMedia(args, tator, row):
    """ Attempts to upload the media in the row to tator
        if already there, skips, but either way returns the
        media element information. """
    vid_dir=os.path.join(args.img_base_dir, row['video_id'])
    global media_list_cache
    try:
        img_file=f"{int(row['frame']):04d}.{args.img_ext}"
    except:
        print(f"Skipping {row}")
        return
    img_path=os.path.join(vid_dir, img_file)
    if args.media_type == "pipeline":
        media_id = row['video_id'].split('_')[0]
        if media_id in media_list_cache:
            return media_list_cache[media_id]
        else:
            result = tator.Media.get(media_id)
            media_list_cache[media_id] = result
            return result
    elif args.media_type == "image":
        desired_name = f"{row['video_id']}_{row['frame']}.{args.img_ext}"
    elif args.media_type == "video":
        desired_name = f"{row['video_id']}.{args.img_ext}"
        img_path = os.path.join(args.img_base_dir, desired_name)
        
    if desired_name in media_list_cache:
        print(f"{time.time()}: In Cache")
        return media_list_cache[desired_name]
    else:
        print(f"{time.time()}: {desired_name}: Not In Cache")
        print(f"Cache = {media_list_cache}")
        media_element_search=tator.Media.filter({"name": desired_name})
        if media_element_search == None:
            print(f"Uploading file...{desired_name}")
            tator.Media.uploadFile(args.media_type_id,
                                   img_path,
                                   progressBars=False,
                                   section=args.section,
                                   fname=desired_name)
            media_element_search=tator.Media.filter({"name": desired_name})
        if media_element_search:
            result = tator.Media.get(media_element_search[0]['id'])
            if result['attributes']['tator_user_sections'] != args.section:
                if args.section:
                    tator.Media.update(result['id'], {'attributes':
                                                      {'tator_user_sections':
                                                       args.section}})
                    print(f"Moving to {args.section}")
            media_list_cache[desired_name] = result
            return result
        else:
            return None



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = pytator.tator.cli_parser(parser)
    parser.add_argument("csvfile", help="test.csv, length.csv, or detect.csv")
    parser.add_argument("--img-base-dir", help="Base Path to media files", required=True)
    parser.add_argument("--img-ext", default="jpg")
    parser.add_argument("--media-type-id", type=int, required=True)
    parser.add_argument("--media-type",
                        type=str,
                        choices=["pipeline", "image","video"],
                        default="image")
    parser.add_argument("--localization-type-id", type=int)
    parser.add_argument("--section", help="Section name to apply")
    parser.add_argument("--train-ini", help="If uploading boxes, this is required to convert species id to a string")
    parser.add_argument("--threshold", type=float, help="Discard boxes less than this value")
    parser.add_argument("--truth-data", type=str, help="Path to annotations.csv to exclude non-truth data")
    parser.add_argument("--species-keyname", type=str,default="Species")
    args = parser.parse_args()
    tator = pytator.Tator(args.url, args.token, args.project)

    signal.signal(signal.SIGINT, exit_func)

    input_data = pd.read_csv(args.csvfile)
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
    input_data["media_id"] = None
    media_map={}
    for idx,row in bar(input_data.iterrows()):
        media_element = uploadMedia(args, tator, row)
        if media_element:
            media_map[media_element['id']] = media_element
            input_data.loc[idx,'media_id'] = media_element['id']
        else:
            print("ERROR: Could not upload.")

    print("Generating localizations...")
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(media_map.keys()))
    for media_id in bar(media_map):
        local_list=[]
        media_data = input_data.loc[input_data['media_id'] == media_id]
        bar2 = progressbar.ProgressBar(redirect_stdout=True, max_value=len(media_data))
        for idx,row in bar2(media_data.iterrows()):
            obj = function_map[mode](args,
                                     tator,
                                     species_names,
                                     truth_data,
                                     media_map,
                                     row)
            if obj:
                local_list.append(obj)

            if len(local_list) == 25:
                try:
                    before=time.time()
                    tator.Localization.addMany(local_list)
                    after=time.time()
                    print(f"Duration={(after-before)*1000}ms")
                except:
                    traceback.print_exc(file=sys.stdout)
                finally:
                    local_list=[]

        try:
            tator.Localization.addMany(local_list)
        except:
            traceback.print_exc(file=sys.stdout)

        # When complete for a given media update the sentinel value
        tator.Media.update(media_id, {"attributes":{"Object Detector Processed": str(datetime.datetime.now())}, "resourcetype": "EntityMediaVideo"})
        tator.Media.update(media_id, {"attributes":{"Object Detector Processed": str(datetime.datetime.now())}, "resourcetype": "EntityMediaImage"})
