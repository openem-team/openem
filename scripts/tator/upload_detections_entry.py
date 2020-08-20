import json
import os
import subprocess
import sys
import urllib.parse

if __name__=="__main__":
    work_dir = os.getenv('TATOR_WORK_DIR')
    work_csv_file = os.path.join(work_dir, 'results.csv')

    url = urllib.parse.urlparse(os.getenv('TATOR_API_SERVICE'))
    host = f"{url.scheme}://{url.netloc}"

    pipeline_args_str = os.getenv('TATOR_PIPELINE_ARGS')
    if pipeline_args_str:
        pipeline_args = json.loads(pipeline_args_str)
    else:
        print("ERROR: No pipeline arguments specified!")
        sys.exit(-1)
    species_attr_name = pipeline_args.get('species_attr_name','Species')
    confidence_attr_name = pipeline_args.get('confidence_attr_name','Confidence')

    args = ["python3", "/scripts/upload_via_tator_py.py",
            "--host", host,
            "--token", os.getenv("TATOR_AUTH_TOKEN"),
            "--csvfile", work_csv_file,
            '--species-attr-name', species_attr_name,
            '--confidence-attr-name', confidence_attr_name]

    cmd = " ".join(args)
    print(f"Upload Detections Command = '{cmd}'")
    p=subprocess.Popen(args)
    p.wait()
    sys.exit(p.returncode)
