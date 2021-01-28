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
    version_id = pipeline_args.get('version_id', None)
    input_version_id = pipeline_args.get('input_version_id', None)
    batch_size = pipeline_args.get('batch_size', 1)
    all_files=os.listdir('/work')

    mode = pipeline_args.get('mode')
    if mode == 'nonvisual':
        media_files=[f"/work/{x}_foo.mp4" for x in os.getenv('TATOR_MEDIA_IDS').split(',')]
    else:
        media_files=[os.path.join('/work',x) for x in all_files if x.endswith('.mp4')]

    optional_args = []
    if input_version_id:
        optional_args.extend(['--input_version_id', str(input_version_id)])
    args = ['python3', '/scripts/tator_tracker.py',
            '--host', os.getenv("TATOR_API_SERVICE").replace('/rest',''),
            '--token', os.getenv("TATOR_AUTH_TOKEN"),
            '--detection-type-id', str(detection_type_id),
            '--tracklet-type-id', str(tracklet_type_id),
            '--version-id', str(version_id),
            '--strategy-config', '/work/strategy.yaml',
            *optional_args,
            *media_files]

    cmd = " ".join(args)
    print(f"Inference Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()
    sys.exit(p.returncode)
    
