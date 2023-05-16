#!/usr/bin/env python3

import os
import pandas as pd
import subprocess
import docker
import json
import pandas as pd
import sys
import shutil
import tarfile
import yaml
import tempfile

import datetime
import traceback
import urllib.request


import tator

if __name__=="__main__":
    work_dir = tempfile.mkdtemp()
    media_ids = os.getenv('TATOR_MEDIA_IDS')
    host = os.getenv("TATOR_API_SERVICE").replace('/rest','')
    token=os.getenv('TATOR_AUTH_TOKEN')
    project_id=os.getenv('TATOR_PROJECT_ID')
    media_ids = [int(m) for m in media_ids.split(',')]
    work_filepath=os.path.join(work_dir, 'work.csv')

    with open('/data/strategy.yaml','r') as fp:
        strategy = yaml.safe_load(fp)
        keep_threshold = strategy.get('keep_threshold', None)
        img_max_side,img_min_side = strategy.get('img_size', None)
        img_ext = strategy.get('img_ext', 'mp4')
        media_type = strategy.get('media_type', 'video')
        batch_size = strategy.get('batch_size', 1)
        graph_pb = strategy.get('graph_pb',None)
        train_ini = strategy.get('train_ini',None)
        species_attr_name = strategy.get('species_attr_name','Species')
        confidence_attr_name = strategy.get('confidence_attr_name','Confidence')
        version_id = strategy.get('version_id', None)
        box_type_id = strategy.get('box_type_id', None)
        sentinel_name = strategy.get('sentinel_name', "Object Detector Processed")

    # Generate work document
    api = tator.get_api(host, token)
    media_list = api.get_media_list_by_id(project_id, {"ids": media_ids})
    media_type_id = media_list[0].type
    media_files = [{'path': f"{m.id}_{m.name}"} for m in media_list if m.attributes.get(sentinel_name, "No") == "No"]
    work_df = pd.DataFrame(columns=['path'],
                           data=media_files)
    work_df.to_csv(f'{work_dir}/work.csv', index=False,header=False)
    if len(work_df) == 0:
        print("No media to process.")
        sys.exit(0)

    # Download network
    os.makedirs("/work/network", exist_ok=True)
    urllib.request.urlretrieve(graph_pb, f"{work_dir}/network/graph.pb")
    urllib.request.urlretrieve(train_ini, f"{work_dir}/network/train_ini")

    args = ['python3', '/scripts/infer.py',
            '--host', host,
            '--token', token,
            '--graph-pb', f'{work_dir}/network/graph.pb',
            '--keep-threshold', str(keep_threshold),
            '--csv-flavor', media_type,
            '--img-base-dir', '/tmp',
            '--img-min-side', str(img_min_side),
            '--img-max-side', str(img_max_side),
            '--img-ext', img_ext,
            '--output-csv', f'{work_dir}/results.csv',
            '--batch-size', str(batch_size),
            f'{work_dir}/work.csv' ]

    if strategy.get('cpu_only',False):
        args.insert(len(args)-1,'--cpu-only')

    cmd = " ".join(args)
    print(f"Inference Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()

    optional_args=[]
    if version_id:
        optional_args = ["--version-id", str(version_id)]

    args = ["python3",
            "/scripts/uploadToTator.py",
            "--host", host,
            "--token", token,
            '--project', str(project_id),
            "--img-base-dir", "/work",
            "--localization-type-id", str(box_type_id),
            "--media-type-id", str(media_type_id),
            '--media-type', 'pipeline',
            *optional_args,
            '--img-ext', img_ext,
            '--species-attr-name', species_attr_name,
            '--confidence-attr-name', confidence_attr_name,
            '--train-ini', f'{work_dir}/network/train.ini',
            f'{work_dir}/results.csv']
    cmd = " ".join(args)
    print(f"Upload Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()

    for m in media_list:
        media_id = m.id
        try:
            # When complete for a given media update the sentinel value
            api.update_media(int(media_id), {'attributes':{sentinel_name: str(datetime.datetime.now())}})
        except Exception as e:
            print(f"Unable to set sentinel attribute {e}")
            traceback.print_exc()

    sys.exit(p.returncode)
