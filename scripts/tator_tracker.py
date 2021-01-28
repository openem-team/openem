#!/usr/bin/env python3

import argparse
import openem
import os
import cv2
import numpy as np
from openem.tracking import *
import json
import sys
import datetime
import tator
from pprint import pprint
from collections import defaultdict

import yaml
import math

import subprocess
import sys

def crop_localization(frame_bgr, localization):
    img_width = frame_bgr.shape[1]
    img_height = frame_bgr.shape[0]
    box_x = round(localization['x'] * img_width)
    box_y = round(localization['y'] * img_height)
    box_width = round(localization['width'] * img_width)
    box_height = round(localization['height'] * img_height)
    img_crop = frame_bgr[box_y:box_y+box_height,box_x:box_x+box_width,:]
    return img_crop

def join_up_iteration(detections, track_ids):
            tracklets = defaultdict(list)
            num_tracklets = np.max(track_ids) + 1
            assert(len(detections) == len(track_ids))
            for d,tid in zip(detections, track_ids):
                tracklets[tid].append(d)
            return tracklets

def extend_tracklets(tracklets, length):
    for track_id,track in tracklets.items():
        if len(track) <= 16:
            continue

        ext_length = min(length,len(track))
        sum_h=0.0
        sum_w=0.0

        track.sort(key=lambda x:x['frame'])

        def restore_det(det):
            det['x'] = det.get('orig_x',det['x'])
            det['y'] = det.get('orig_y',det['y'])
            det['width'] = det.get('orig_w',det['width'])
            det['height'] = det.get('orig_h',det['height'])
            det['orig_x'] = det['x']
            det['orig_y'] = det['y']
            det['orig_w'] = det['width']
            det['orig_h'] = det['height']
        restore_det(track[0])
        restore_det(track[-1])

        for d in track:
            sum_h += d['height']
            sum_w += d['width']
        angle,vel,comps = track_vel(track)
        vel_x = comps[0]
        vel_y = comps[1]
        avg_h = sum_h / len(track)
        avg_w = sum_w / len(track)
        new_x = min(1,max(0,track[-1]['x']+(vel_x*ext_length)))
        new_y = min(1,max(0,track[-1]['y']+(vel_y*ext_length)))
        old_x = min(1,max(0,track[0]['x']-(vel_x*ext_length)))
        old_y = min(1,max(0,track[0]['y']-(vel_y*ext_length)))


        min_x = min(track[-1]['x'],new_x)
        min_y = min(track[-1]['y'],new_y)
        if min_x > 0 and min_y > 0:
            track[-1]['x'] = min_x
            track[-1]['y'] = min_y
            track[-1]['width'] = min(max(0,abs(new_x-track[-1]['x'])+avg_w),1)
            track[-1]['height'] = min(max(0,abs(new_x-track[-1]['y'])+avg_h),1)
        else:
            track[-1]['width'] = 0
            track[-1]['height'] = 0


        min_x = min(track[0]['x'],old_x)
        min_y = min(track[0]['y'],old_y)
        if min_x > 0 and min_y > 0:
            track[0]['x'] = min(max(0,min_x),1)
            track[0]['y'] = min(max(0,min_y),1)
            track[0]['width'] = min(max(abs(old_x-track[0]['x'])+avg_w,0),1)
            track[0]['height'] = min(max(abs(old_x-track[0]['y'])+avg_h,0),1)
        else:
            track[0]['width'] = 0
            track[0]['height'] = 0
    return tracklets


def split_tracklets(tracklets):
    track_ids=[]
    detections=[]
    for track_id,track in tracklets.items():
        for d in track:
            track_ids.append(track_id)
            detections.append(d)
    return detections,track_ids

def trim_tracklets(detections, track_ids, max_length):
    tracklets = join_up_iteration(detections, track_ids)
    next_track_id = 1
    new_tracklets = {}
    for track_id,detections in tracklets.items():
        new_track_count=math.ceil(len(detections)/max_length)
        for i in range(new_track_count):
            start=max_length*i
            end=max_length+(max_length*i)
            new_tracklets[next_track_id] = detections[start:end]
            next_track_id += 1
    detections, track_ids = split_tracklets(new_tracklets)
    track_ids = renumber_track_ids(track_ids)
    return detections, track_ids


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    tator.get_parser(parser)
    parser.add_argument("--detection-type-id", type=int, required=True)
    parser.add_argument("--tracklet-type-id", type=int, required=True)
    parser.add_argument("--version-id", type=int)
    parser.add_argument("--input-version-id", type=int)
    parser.add_argument("--strategy-config", type=str)
    parser.add_argument("--dry-run", action='store_true')
    parser.add_argument('media_files', type=str, nargs='*')
    args = parser.parse_args()

    # Weight methods
    methods = ['hybrid', 'iou', 'iou-motion']

    # Weight methods that require the video
    visual_methods = ['hybrid']

    api = tator.get_api(args.host, args.token)
    detection_type = api.get_localization_type(args.detection_type_id)
    project = detection_type.project
    version_id = args.version_id


    default_strategy = {"method": "hybrid",
                        "frame-diffs": [1,2,4,8,16,32,64,128,256],
                        "args": {},
                        "extension": {'method' : None},
                        "max-length": {},
                        "min-length": 0}

    if args.strategy_config:
        strategy = {**default_strategy}
        with open(args.strategy_config, "r") as strategy_file:
            strategy.update(yaml.load(strategy_file))
    else:
        strategy = default_strategy

    if strategy['method'] == 'hybrid':
        model_file = strategy['args']['model_file']
        batch_size = strategy['args'].get('batch_size', 4)
        comparator=FeaturesComparator(model_file)
        #extractor=FeaturesExtractor(args.model_file)


    class_method = strategy.get('class-method',None)
    classify_function = None
    classify_args = {}
    if class_method:
        pip_package=class_method.get('pip',None)
        if pip_package:
            p = subprocess.run([sys.executable,
                                "-m",
                                "pip",
                                "install",
                                pip_package])
            print("Finished process.", flush=True)
        function_name = class_method.get('function',None)
        classify_args = class_method.get('args',None)
        names = function_name.split('.')
        module = __import__(names[0])
        for name in names[1:-1]:
            module = getattr(module,name)
        classify_function = getattr(module,names[-1])


    print("Strategy: ", flush=True)
    pprint(strategy)
    print(args.media_files, flush=True)

    optional_fetch_args = {}
    if args.input_version_id:
        optional_fetch_args['version'] = args.input_version_id
    for media_file in args.media_files:
        localizations_by_frame = {}
        comps=os.path.splitext(os.path.basename(media_file))[0]
        media_id=comps.split('_')[0]
        localizations = api.get_localization_list(project,
                                                  type=args.detection_type_id,
                                                  media_id=[media_id],
                                                  **optional_fetch_args)

        localizations = [l.to_dict() for l in localizations]
        if len(localizations) == 0:
            print(f"No localizations present in media {media_file}", flush=True)
            continue
        print(f"Processing {len(localizations)} detections", flush=True)
        # Group by localizations by frame
        for lid, local in enumerate(localizations):
            frame = local['frame']
            if frame in localizations_by_frame:
                localizations_by_frame[frame].append(local)
            else:
                localizations_by_frame[frame] = [local]

        detections=[]
        track_ids=[]
        track_id=1

        media = api.get_media(media_id)
        media_shape = (media.height, media.width)
        fps = media.fps

        if strategy['method'] in visual_methods:
            vid=cv2.VideoCapture(media_file)
            ok=True
            frame = 0
            while ok:
                ok,frame_bgr = vid.read()
                if frame in localizations_by_frame:
                    for l in localizations_by_frame[frame]:
                        if strategy['method'] == 'hybrid':
                            l['bgr'] = crop_localization(frame_bgr, l)
                        if l['attributes']['Confidence'] < 0.50:
                            continue
                        detections.append(l)
                        track_ids.append(track_id)
                        track_id += 1
                frame+=1
        else:
            # The method is analytical on the detections coordinates
            # and does not require processing the video
            for frame,frame_detections in localizations_by_frame.items():
                for det in frame_detections:
                    detections.append(det)
                    track_ids.append(track_id)
                    track_id += 1

        print("Loaded all detections", flush=True)

        track_ids = renumber_track_ids(track_ids)

        if strategy['method'] == 'hybrid':
            weights_strategy = HybridWeights(comparator,
                                             None,
                                             None,
                                             media_shape,
                                             fps,
                                             0.0,
                                             batch_size)
        elif strategy['method'] == 'iou':
            weights_strategy = IoUWeights(media_shape, **strategy['args'])
        elif strategy['method'] == 'iou-motion':
            weights_strategy = IoUMotionWeights(media_shape, **strategy['args'])
        # Generate localization bgr based on grouped localizations
        for x in strategy['frame-diffs']:
            print(f"Started {x}", flush=True)
            detections, track_ids, pairs, weights, is_cut, constraints = join_tracklets(
                detections,
                track_ids,
                x,
                weights_strategy)

            if x in strategy['max-length']:
                trim_to = strategy['max-length'][x]
                print(f"Trimming track to max length of {trim_to}")
                detections, track_ids = trim_tracklets(detections, track_ids, trim_to)
            _,det_counts_per_track=np.unique(track_ids,return_counts=True)
            print(f"frame-diff {x}: {len(detections)} to {len(det_counts_per_track)}", flush=True)

            if x > 1 and strategy['extension']['method'] == 'linear-motion':
                ext_frames=x
                print(f"Extending by linear motion, {ext_frames}")
                tracklets = join_up_iteration(detections,track_ids)
                tracklets = extend_tracklets(tracklets, ext_frames)
                detections, track_ids = split_tracklets(tracklets)

        # Now we make new track objects based on the result
        # from the graph solver
        # [ detection, detection, detection, ...]
        # [ track#, track#, track#,...]
        # [ 133, 33, 13, 133,]
        # [ 0,0,1,1]
        # TODO: Handle is_cut?
        def join_up_final(detections, track_ids):
            tracklets = defaultdict(list)
            num_tracklets = np.max(track_ids) + 1
            assert(len(detections) == len(track_ids))
            for d,tid in zip(detections, track_ids):
                tracklets[tid].append(d)
            return tracklets

        def make_object(track):
            track.sort(key=lambda x:x['frame'])
            if classify_function:
                valid,attrs = classify_function(media.to_dict(),
                                                track,
                                                **classify_args)
            elif len(track) >= strategy['min-length']:
                valid = True
                attrs = {}
            else:
                valid = False
                attrs = {}
            if valid:
                obj={"type": args.tracklet_type_id,
                     "media_ids": [int(media_id)],
                     "localization_ids": [x['id'] for x in track],
                     **attrs,
                     "version": version_id}
                return obj
            else:
                return None

        tracklets = join_up_final(detections, track_ids)
        new_objs=[make_object(tracklet) for tracklet in tracklets.values()]
        new_objs=[x for x in new_objs if x is not None]
        print(f"New objects = {len(new_objs)}")
        with open(f"/work/{media_id}.json", "w") as f:
            json.dump(new_objs,f)
        if not args.dry_run:
            for response in tator.util.chunked_create(api.create_state_list,project,
                                                      state_spec=new_objs):
                pass
            try:
                api.update_media(int(media_id), {"attributes":{"Tracklet Generator Processed": str(datetime.datetime.now())}})
            except:
                print("WARNING: Unable to set 'Tracklet Generator Processed' attribute")
