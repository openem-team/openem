#!/usr/bin/env python3
''' This script is exepcted to be executed after detections have been made in the openem pipeline
'''

import os
import subprocess
import sys

if __name__=="__main__":
    work_dir = os.getenv('TATOR_WORK_DIR')
    work_csv_file = os.path.join(work_dir, 'work.csv')

    args = ["python3", "/scripts/iou_correlation_filter_tracker.py",
            "--url", os.getenv("TATOR_API_SERVICE"),
            "--token", os.getenv("TATOR_AUTH_TOKEN"),
            "--csv", work_csv_file,
            "--max-coast-age", "5",
            "--association-threshold", "0.4"]

    cmd = " ".join(args)
    print(f"Track Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()
    sys.exit(p.returncode)
    
