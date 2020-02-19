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

COPY graph.pb /network/graph.pb
COPY train.ini /network/train.ini 


"""
if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-pb", required=True)
    parser.add_argument("--train-ini", required=True)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("image_tag")
    args = parser.parse_args()
    
    temp_dir = tempfile.mkdtemp()
    with open(os.path.join(temp_dir, "Dockerfile"), 'w') as docker_file:
        docker_file.write(DOCKER_FILE_TEMPLATE)
    shutil.copy(args.graph_pb, os.path.join(temp_dir, "graph.pb"))
    shutil.copy(args.train_ini, os.path.join(temp_dir, "train.ini"))
    #print(temp_dir)
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
