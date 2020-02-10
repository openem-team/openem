#!/usr/bin/env python3

import pytator
import json
import os
import sys
import time
import subprocess

if __name__ == '__main__':
    rest_svc = os.getenv('TATOR_API_SERVICE')
    work_dir = os.getenv('TATOR_WORK_DIR')
    token = os.getenv('TATOR_AUTH_TOKEN')
    project_id = os.getenv('TATOR_PROJECT_ID')
    pipeline_args_str = os.getenv('TATOR_PIPELINE_ARGS')
    if pipeline_args_str:
        pipeline_args = json.loads(pipeline_args_str)
    else:
        print("ERROR: No pipeline arguments specified!")
        sys.exit(-1)
    box_type_id = pipeline_args['type_id']
    img_ext = pipeline_args.get('img_ext', None)
    media_type = pipeline_args.get('media_type', None)
    
    args = ["python3",
            "/scripts/uploadToTator.py",
            "--url", rest_svc,
            "--project", str(project_id),
            "--token", token
            "--img-base-dir", "/work",
            "--localization-type-id", str(box_type_id),
            "--media-type-id", 0,
            '--media-type', media_type,
            '--img-ext', image_ext,
            '--train-ini', '/work/train.ini',
            '/work/results.csv']
    cmd = " ".join(args)
    print(f"Inference Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()
    return p.returncode

            
