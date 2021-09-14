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

    tracklet_type_id = pipeline_args.get('tracklet_type_id')
    version_id = pipeline_args.get('version_id')
    attribute_name = pipeline_args.get('attribute_name')
    batch_size = pipeline_args.get('batch_size', 1)
    all_files=os.listdir('/work')
    media_ids=os.getenv('TATOR_MEDIA_IDS').split(',')

    args = ['python3', '/scripts/measure.py',
            '--host', os.getenv("TATOR_API_SERVICE").replace('/rest',''),
            '--token', os.getenv("TATOR_AUTH_TOKEN"),
            '--tracklet-type-id', str(tracklet_type_id),
            '--version-id', str(version_id),
            '--attribute-name', str(attribute_name),
            '--strategy-config', '/work/strategy.yaml',
            *media_ids]

    cmd = " ".join(args)
    print(f"Inference Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()
    sys.exit(p.returncode)
    
