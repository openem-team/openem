#!/usr/bin/env python3

import pytator
import os
import sys
import json
import pandas as pd
import requests

if __name__ == '__main__':
    media_ids = os.getenv('TATOR_MEDIA_IDS')
    print(f"processing = {media_ids)")
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

    # Download graph file
    graph_url = pipeline_args.get('graph_url')
    train_ini= pipeline_args.get('train_ini')
    graph_response = requests.get(graph_url)
    if graph_response.status_code == 200:
        with open('/work/graph.pb', 'wb') as graph_fp:
            graph_fp.write(graph_response.content)
    else:
        print(f"Error downloading {graph_url}")

    train_response = requests.get(train_ini)
    if train_response.status_code == 200:
        with open('/work/train.ini', 'wb') as train_fp:
            train_fp.write(train_response.content)
    else:
        print(f"Error downloading {train_ini}")

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
