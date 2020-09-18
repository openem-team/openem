#!/usr/bin/env python3

import pytator
import json
import os
import sys
import time
import subprocess
import sys

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
    image_ext = pipeline_args.get('img_ext', None)
    species_attr_name = pipeline_args.get('species_attr_name','Species')
    confidence_attr_name = pipeline_args.get('confidence_attr_name','Confidence')
    optional_args=[]
    version_number = pipeline_args.get('version_number', None)
    if version_number:
        optional_args.extend(['--version-number', str(version_number)])

    args = ["python3",
            "/scripts/uploadToTator.py",
            "--url", rest_svc,
            "--project", str(project_id),
            "--token", token,
            "--img-base-dir", "/work",
            "--localization-type-id", str(box_type_id),
            "--media-type-id", str(0),
            '--media-type', media_type,
            *optional_args,
            '--img-ext', image_ext,
            '--species-attr-name', species_attr_name,
            '--confidence-attr-name', confidence_attr_name,
            '--train-ini', '/work/network/train.ini',
            '/work/results.csv']
    cmd = " ".join(args)
    print(f"Upload Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()
    sys.exit(p.returncode)

            
