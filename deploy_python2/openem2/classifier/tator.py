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
import time

import docker
import tator

from . import thumbnail_classifier

def _extract_tracks(api, media, trackTypeId, **kwargs):
    print(f"Extracting tracks from '{media.name}'")
    temp_dir = tempfile.mkdtemp()
    local_media = os.path.join(temp_dir, media.name)
    for _ in tator.download_media(api,
                                  media,
                                  local_media):
        pass
    assert os.path.exists(local_media)

    tracks = api.get_state_list(media.project,
                                media_id=[media.id],
                                type=trackTypeId)
    tracks = [t.to_dict() for t in tracks]

    by_frame = defaultdict(lambda: [])
    for track in tracks:
        localizations = api.get_localization_list_by_id(
            media.project,
            {"ids": track['localizations']})
        localizations = [l.to_dict() for l in localizations]
        for l in localizations:
            l['track_id'] = track['id']
            by_frame[l['frame']].append(l)
        track['local_objs'] = localizations

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

def run_tracker(api,
                project,
                media_ids,
                model_dir,
                strategy):
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
    classifier = thumbnail_classifier.EnsembleClassifier(
        model_files,
        **strategy['ensemble_config'])

    track_type_id = strategy['tator']['track_type_id']
    medias = api.get_media_list_by_id(project, {"ids": media_ids})
    for media in medias:
        tracks = _extract_tracks(api, media, track_type_id)
        processed_time = media.attributes.get("Track Classification Processed",
                                              "No")
        if processed_time != "No":
            print(f"Already processed '{media.name}'")
            continue

        for track_id,thumbnails in tracks.items():
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
            api.update_state(track_id, update)
            print(f"{track_id}: {label} {track_entropy}")

        api.update_media(media.id,
                         {'attributes':
                          {"Track Classification Processed":
                           datetime.datetime.now().isoformat()}})

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

    project = api.get_state_type(strategy['tator']['track_type_id']).project

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
