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
        data_image = strategy.get('data_image',None)
        species_attr_name = strategy.get('species_attr_name','Species')
        confidence_attr_name = strategy.get('confidence_attr_name','Confidence')
        version_id = strategy.get('version_id', None)
        box_type_id = strategy.get('box_type_id', None)

    # Generate work document
    api = tator.get_api(host, token)
    media_list = api.get_media_list_by_id(project_id, media_ids)
    media_type_id = media_list[0].meta
    media_files = [{'path': f"{m.id}_{m.name}"} for m in media_list if m.attributes.get("Object Detector Processed", "No") == "No"]
    work_df = pd.DataFrame(columns=['path'],
                           data=media_files)
    work_df.to_csv(f'{work_dir}/work.csv', index=False,header=False)
    if len(work_df) == 0:
        print("No media to process.")
        sys.exit(0)

    # Download network
    client=docker.from_env()
    image=client.images.pull(data_image)
    container=client.containers.create(data_image)
    bits, stats = container.get_archive("/network")
    network_tar = os.path.join(work_dir, "network.tar")
    with open(network_tar, 'wb') as tar_file:
        for chunk in bits:
            tar_file.write(chunk)

    with tarfile.TarFile(network_tar) as tar_file:
        print(tar_file.getmembers())
        tar_file.extract("network/graph.pb", work_dir)
        tar_file.extract("network/train.ini", work_dir)

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

    args = ["python3",
            "/scripts/uploadToTator.py",
            "--host", host,
            "--token", token,
            "--img-base-dir", "/work",
            "--localization-type-id", str(box_type_id),
            "--media-type-id", str(media_type_id),
            '--media-type', media_type,
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

    sys.exit(p.returncode)
