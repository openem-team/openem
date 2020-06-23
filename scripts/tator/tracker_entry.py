#!/usr/bin/env python3

import os
import pandas as pd
import subprocess
import json
import sys
import shutil

if __name__=="__main__":
    work_dir = os.getenv('TATOR_WORK_DIR')
    pipeline_args_str = os.getenv('TATOR_PIPELINE_ARGS')
    if pipeline_args_str:
        pipeline_args = json.loads(pipeline_args_str)
    else:
        pipeline_args = {}

    detection_type_id = pipeline_args.get('detection_type_id', None)
    tracklet_type_id = pipeline_args.get('tracklet_type_id', None)
    version_number = pipeline_args.get('version_number', None)
    batch_size = pipeline_args.get('batch_size', 1)
    all_files=os.listdir('/work')
    media_files=[os.path.join('/work',x) for x in all_files if x.endswith('.mp4')]

    args = ['python3', '/scripts/tator_tracker.py',
            '--url', os.getenv("TATOR_API_SERVICE"),
            '--token', os.getenv("TATOR_AUTH_TOKEN"),
            '--project', os.getenv('TATOR_PROJECT_ID'),
            '--model-file', '/work/network/graph.pb',
            '--detection-type-id', str(detection_type_id),
            '--tracklet-type-id', str(tracklet_type_id),
            '--version-number', str(version_number),
            '--batch-size', str(batch_size),
            *media_files]

    cmd = " ".join(args)
    print(f"Inference Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()
    sys.exit(p.returncode)
    
