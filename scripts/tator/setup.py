#!/usr/bin/env python3

import pytator
import docker
import tarfile
import os
import sys
import json
import pandas as pd
import requests
import time

if __name__ == '__main__':
    media_ids = os.getenv('TATOR_MEDIA_IDS')
    print(f"processing = {media_ids})")
    media_ids = [int(m) for m in media_ids.split(',')]
    rest_svc = os.getenv('TATOR_API_SERVICE')
    work_dir = os.getenv('TATOR_WORK_DIR')
    token=os.getenv('TATOR_AUTH_TOKEN')
    project_id=os.getenv('TATOR_PROJECT_ID')
    pipeline_args_str = os.getenv('TATOR_PIPELINE_ARGS')
    if pipeline_args_str:
        pipeline_args = json.loads(pipeline_args_str)
    else:
        pipeline_args = {}
    tator=pytator.Tator(rest_svc, token, project_id)
    
    work_filepath=os.path.join(work_dir, "work.csv")
    try:
        os.remove(work_filepath)
    except:
        pass

    # Download the network coefficients
    # Image stores coeffients in "/network" folder
    client=docker.from_env()
    image=client.images.pull(pipeline_args['data_image'])
    container=client.containers.create(pipeline_args['data_image'])
    bits, stats = container.get_archive("/network")
    network_tar = os.path.join(work_dir, "network.tar") 
    with open(network_tar, 'wb') as tar_file:
        for chunk in bits:
            tar_file.write(chunk)

    with tarfile.TarFile(network_tar) as tar_file:
        print(tar_file.getmembers())
        tar_file.extract("network/graph.pb", work_dir)
        tar_file.extract("network/train.ini", work_dir)

    container.remove()
    # First write CSV header
    cols=['media']
    work_frame=pd.DataFrame(columns=cols)
    work_frame.to_csv(work_filepath, index=False)

    print(f"Starting on {work_filepath}")
    for media_id in media_ids:
        media = tator.Media.get(media_id)
        media_unique_name = f"{media['id']}_{media['name']}"
        media_filepath = os.path.join(work_dir,media_unique_name)
        data={'media': media_filepath}
        print(f"Downloading {media['name']} to {media_filepath}")
        tator.Media.downloadFile(media, media_filepath)
        work_frame=pd.DataFrame(columns=cols, data=[data])
        work_frame.to_csv(work_filepath, index=False, header=False, mode='a')
