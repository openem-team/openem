#!/usr/bin/env python3

""" Script to generate network images 

With embedded network files.

"""

import argparse
import subprocess
import tempfile
import shutil
import os

# Use alpine because we need docker create to work
DOCKER_FILE_TEMPLATE="""
FROM alpine:3.7 AS images

RUN mkdir /network
"""
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--publish", action="store_true",
                        help="If supplied pushes to image repo")
    parser.add_argument("--image-tag", help="Name of image to build/publish")
    parser.add_argument("models", nargs="+",
                        help="One or more models to encorporate into image.")
    args = parser.parse_args()
    
    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(temp_dir, "Dockerfile"), 'w') as docker_file:
        docker_file.write(DOCKER_FILE_TEMPLATE)
        for idx,model in enumerate(args.models):
            ext = os.path.splitext(model)[1]
            shutil.copy(model, os.path.join(temp_dir, f"weights_{idx}{ext}"))
            docker_file.write(f"COPY weights_{idx}{ext} /network/weights_{idx}{ext}\n")
    
    with open(os.path.join(temp_dir, "Dockerfile"), 'r') as docker_file:
        print(docker_file.read())
    cargs=['docker',
          'build',
          temp_dir,
          '-f',
          os.path.join(temp_dir, 'Dockerfile'),
          '-t',
          args.image_tag]
    print(" ".join(cargs))
    c = subprocess.Popen(cargs)
    c.wait()
    shutil.rmtree(temp_dir)

    if args.publish:
        cargs=['docker',
              'push',
              args.image_tag]
        c = subprocess.Popen(cargs)
        c.wait()
