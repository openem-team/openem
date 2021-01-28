#!/usr/bin/env python3

import tator
import docker
import tarfile
import os
import sys
import json
import pandas as pd
import requests
import time
import math

if __name__ == '__main__':
    media_ids = os.getenv('TATOR_MEDIA_IDS')
    print(f"processing = {media_ids})")
    media_ids = [int(m) for m in media_ids.split(',')]
    host = os.getenv("TATOR_API_SERVICE").replace('/rest','')
    work_dir = os.getenv('TATOR_WORK_DIR')
    token=os.getenv('TATOR_AUTH_TOKEN')
    project_id=os.getenv('TATOR_PROJECT_ID')
    pipeline_args_str = os.getenv('TATOR_PIPELINE_ARGS')
    if pipeline_args_str:
        pipeline_args = json.loads(pipeline_args_str)
    else:
        pipeline_args = {}
    api=tator.get_api(host, token)
    
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

    media_elements = api.get_media_list(project_id,
                                        media_id=media_ids)
    print(f"Starting on {work_filepath}")
    for media_id,media in zip(media_ids,media_elements):
        media_unique_name = f"{media.id}_{media.name}"
        media_filepath = os.path.join(work_dir,media_unique_name)
        data={'media': media_filepath}
        print(f"Downloading {media.name} to {media_filepath}")
        for _,_ in tator.download_media(api,media, media_filepath):
            pass
        if not os.path.exists(media_filepath):
            print("File did not download!")
            sys.exit(255)
        work_frame=pd.DataFrame(columns=cols, data=[data])
        work_frame.to_csv(work_filepath, index=False, header=False, mode='a')
