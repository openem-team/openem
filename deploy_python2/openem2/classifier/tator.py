#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict
import cv2
import datetime
import numpy as np
from pprint import pprint
import shutil
import tarfile
import tempfile
import yaml
import subprocess
import time
import sys

import docker
import tator

from . import thumbnail_classifier

def _extract_tracks(api, media, trackTypeId, **kwargs):
    print(f"Extracting tracks from '{media.name}'")
    mode = kwargs.get('mode', 'state')
    print(f"Extracing mode = {mode}")
    if mode == 'state':
        tracks = api.get_state_list(media.project,
                                media_id=[media.id],
                                type=trackTypeId)
    elif mode == 'localization':
        # call these tracks even though its a track of one detection
        tracks = api.get_localization_list(media.project,
                                           media_id=[media.id],
                                           type=trackTypeId)
    if len(tracks) == 0:
        return None
    temp_dir = tempfile.mkdtemp()
    local_media = os.path.join(temp_dir, media.name)
    for _ in tator.download_media(api,
                                  media,
                                  local_media):
        pass
    assert os.path.exists(local_media)

    tracks = [t.to_dict() for t in tracks]
    by_frame = defaultdict(lambda: [])
    if mode == 'state':
        for track in tracks:
            localizations = api.get_localization_list_by_id(
                media.project,
                {"ids": track['localizations']})
            localizations = [l.to_dict() for l in localizations]
            for l in localizations:
                l['track_id'] = track['id']
                by_frame[l['frame']].append(l)
    elif mode == 'localization':
        for local in tracks:
            local['track_id'] = local['id'] # just use local id
            by_frame[local['frame']].append(local)
    max_frame = max(list(by_frame.keys()))
    print(f"Processing {len(tracks)} up to frame {max_frame}")
    reader = cv2.VideoCapture(local_media)
    frame_idx = 0
    ok = True
    track_thumbs = defaultdict(lambda: [])
    while frame_idx <= max_frame and ok is True:
        ok,frame_bgr = reader.read()
        if ok is False:
            print(f"Couldn't read frame {frame_idx}, '{local_media}'")
            break
        media_height,media_width,_ = frame_bgr.shape
        if frame_idx in by_frame:
            for localization in by_frame[frame_idx]:
                if localization.get('x') < 0:
                    localization['x'] = 0
                if localization.get('y') < 0:
                    localization['y'] = 0

                thumb_height = int(media_height * localization['height'])
                thumb_width = int(media_width * localization['width'])
                thumb_y = int(media_height * localization['y'])
                thumb_x = int(media_width * localization['x'])
                # Check for annotations extending off screen
                if thumb_x + thumb_width > media_width:
                    thumb_width = media_width - thumb_x
                if thumb_y + thumb_height > media_height:
                    thumb_height = media_height - thumb_y

                # crop and save in high quality
                thumbnail = frame_bgr[thumb_y:thumb_y+thumb_height,
                                      thumb_x:thumb_x+thumb_width,
                                      :];
                track_thumbs[localization['track_id']].append(thumbnail)

        frame_idx+=1
    del reader
    _try_delete(temp_dir)
    return track_thumbs

def _try_delete(directory):
    deleted = False
    while deleted is False:
        try:
            shutil.rmtree(directory)
        except:
            time.sleep(1)
        deleted = True

def _preprocess_thumbnail(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image.astype(np.float64)
    image /= 255.0
    return image

def _safe_retry(function, *args,**kwargs):
    count = kwargs.get('_retry_count', 5)
    if '_retry_count' in kwargs:
        del kwargs['_retry_count']
    complete = False
    fail_count = 0
    while complete is False:
        try:
            return_value = function(*args,**kwargs)
            complete = True
            return return_value
        except Exception as e:
            fail_count += 1
            if fail_count > count:
                raise e



def run_tracker(api,
                project,
                media_ids,
                model_dir,
                strategy):
    filter_method = strategy.get('filter-method',None)
    filter_function = None
    filter_args = {}
    if filter_method:
        pip_package=filter_method.get('pip',None)
        if pip_package:
            p = subprocess.run([sys.executable,
                                "-m",
                                "pip",
                                "install",
                                pip_package])
            print("Finished process.", flush=True)
        function_name = filter_method.get('function',None)
        filter_args = filter_method.get('args',{})
        names = function_name.split('.')
        module = __import__(names[0])
        for name in names[1:-1]:
            module = getattr(module,name)
        filter_function = getattr(module,names[-1])

    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
    classifier = thumbnail_classifier.EnsembleClassifier(
        model_files,
        **strategy['ensemble_config'])

    track_type_id = strategy['tator']['track_type_id']
    extract_mode = strategy['tator'].get('extract_mode', 'state')
    assert extract_mode in ['state','localization']
    update_mode = strategy['tator'].get('update_mode', 'patch')
    assert update_mode in ['patch', 'post']
    update_type_id = strategy['tator'].get('update_type_id',None)
    update_match = strategy['tator'].get('update_match',None)
    version_id = strategy['tator'].get('version_id', None)

    medias = api.get_media_list_by_id(project, {"ids": media_ids})
    for media in medias:
        processed_time = media.attributes.get("Track Classification Processed",
                                              "No")
        if processed_time != "No" and not strategy['tator'].get('force', False):
            print(f"Already processed '{media.name}'")
            continue

        tracks = _extract_tracks(api, media, track_type_id, mode=extract_mode)

        if tracks is None:
            print(f"No tracks to process")
            _safe_retry(api.update_media,
                        media.id,
                        {'attributes':
                            {"Track Classification Processed": datetime.datetime.now().isoformat()}
                        }
                    )
            continue

        for track_id,thumbnails in tracks.items():
            # Run pre-processing filter first
            label, winner, track_entropy = None, -1, None
            if filter_function:
                label,track_entropy = filter_function(api, project, track_id, thumbnails, **filter_args)
            if label is None:
                # Convert each thumbnail to RGB and scale to 0-1
                thumbnails = [_preprocess_thumbnail(t) for t in thumbnails]
                detection_scores, entropy = classifier.run_track(thumbnails)
                label, winner, track_entropy = classifier.process_track_results(
                    detection_scores,
                    entropy,
                    **strategy['track_params'])
            update = {'attributes':
                      {strategy['tator']['label_attribute']: label,
                      'Entropy': track_entropy}}

            # if supplied, update version
            if version_id:
                update.update({"version": version_id})
            # If in post mode, check to see if we are only posting if the given a
            # certain label.
            if update_mode == 'post':
                if update_match:
                    post_new_state = (update_match == label)
                else:
                    post_new_state = True
            else:
                post_new_state = False

            print(f"update_mode = {update_mode}, winner = {winner}, post_new_state={post_new_state}")
            if update_mode == 'patch' or not post_new_state:
                if extract_mode == 'state':
                    function = api.update_state
                elif extract_mode == 'localization':
                    function = api.update_localization
                _safe_retry(function,track_id, update)
            elif update_mode == 'post':
                update.update({"type": update_type_id})
                if extract_mode == 'state':
                    assert False # not supported
                elif extract_mode == 'localization':
                    print("Posting new object!")
                    existing_obj = api.get_localization(track_id)
                    # Copy in existing positional information
                    update['x'] = existing_obj.x
                    update['y'] = existing_obj.y
                    update['width'] = existing_obj.width
                    update['height'] = existing_obj.height
                    update['media_id'] = existing_obj.media
                    update['frame'] = existing_obj.frame
                    update.update(update['attributes'])
                    del update['attributes']
                    _safe_retry(api.create_localization_list,project, [update])

            print(f"{track_id}: {label} {track_entropy}")

        _safe_retry(api.update_media,
                    media.id,
                    {'attributes':
                        {"Track Classification Processed": datetime.datetime.now().isoformat()}
                    }
                )

def main():
    """ Invokes the thumbnail classifier on a media in tator """
    parser = argparse.ArgumentParser(description=__doc__)
    parser = tator.get_parser(parser)
    parser.add_argument("--strategy",
                        help="path to strategy file",
                        required=True)
    parser.add_argument("media_ids", nargs="*",help="List of media ids")
    args = parser.parse_args()
    print(args)

    host = os.getenv("TATOR_API_SERVICE")
    if host:
        host = host.replace('/rest', '')
    token = os.getenv("TATOR_AUTH_TOKEN")
    media_ids = None
    media_ids_str = os.getenv("TATOR_MEDIA_IDS")
    if media_ids_str:
        media_ids = [int(x) for x in media_ids_str.split(',')]

    # Override ENV with CLI
    if host is None and args.host:
        host = args.host
    if args.token:
        token = args.token
    if args.media_ids:
        media_ids = args.media_ids


    api = tator.get_api(host, token)

    with open(args.strategy) as fp:
        strategy = yaml.safe_load(fp)

    data_image = strategy['data_image']
    print("Strategy:")
    pprint(strategy)

    extract_mode = strategy['tator'].get('extract_mode', 'state')
    assert extract_mode in ['state','localization']
    if extract_mode == 'state':
        project = _safe_retry(api.get_state_type,strategy['tator']['track_type_id']).project
    else:
        project = _safe_retry(api.get_localization_type,strategy['tator']['track_type_id']).project
    # Download the network files from docker
    network_dir = tempfile.mkdtemp()
    client=docker.from_env()
    image=client.images.pull(data_image)
    container=client.containers.create(data_image)
    bits, stats = container.get_archive("/network")
    network_tar = os.path.join(network_dir, "network.tar")
    with open(network_tar, 'wb') as tar_file:
        for chunk in bits:
            tar_file.write(chunk)
    container.remove()

    with tarfile.TarFile(network_tar) as tar_file:
        members = tar_file.getmembers()
        for member in members:
            tar_file.extract(member, network_dir)
    # Remove tar file
    os.remove(network_tar)

    run_tracker(api,
                project,
                media_ids,
                os.path.join(network_dir, 'network'),
                strategy)

    shutil.rmtree(network_dir)

if __name__=="__main__":
    main()
