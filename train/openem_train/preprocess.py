__copyright__ = "Copyright (C) 2018 CVision AI."
__license__ = "GPLv3"
# This file is part of OpenEM, released under GPLv3.
# OpenEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

"""Functions for preprocessing training data.
"""

import os
import sys
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing.sharedctypes import Value
from multiprocessing.context import TimeoutError
from ctypes import c_longlong
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from functools import partial
import pandas as pd
import scipy.misc
import skimage
from cv2 import VideoCapture
from cv2 import imwrite
import progressbar

# Current value for multi-processed loops
current=Value(c_longlong, 0)

def _extract_images(vid, train_imgs_dir):
    """Extracts images from a single video.

    # Arguments
        vid: Path to video.
        train_imgs_dir: Path to output images.

    # Returns
        Tuple containing video ID and number of frames.
    """
    vid_id, _ = os.path.splitext(os.path.basename(vid))
    img_dir = os.path.join(train_imgs_dir, vid_id)
    os.makedirs(img_dir, exist_ok=True)
    reader = VideoCapture(vid)
    frame = 0
    while reader.isOpened():
        ret, img = reader.read()
        if not ret:
            break
        img_path = os.path.join(img_dir, '{:04}.jpg'.format(frame))
        imwrite(img_path, img)
        frame += 1
    current.value += 1
    return (vid_id, frame)

def extract_images(config):
    """Extracts images from all videos.

    # Arguments
        config: ConfigInterface object.
    """

    # Create directories to store images.
    os.makedirs(config.train_imgs_dir(), exist_ok=True)

    # Make a pool to convert videos.
    pool = Pool(24)
    current.value = 0

    # Start converting images.
    func = partial(_extract_images, train_imgs_dir = config.train_imgs_dir())
    vid_list=config.train_vids()
    bar = progressbar.ProgressBar(max_value=len(vid_list),
                                  redirect_stdout=True,
                                  redirect_stderr=True)
    ar = pool.map_async(func, vid_list)
    vid_frames = None
    while vid_frames == None:
        try:
            vid_frames = ar.get(2)
        except TimeoutError as te:
            pass
        finally:
            bar.update(current.value)


    # Record total number of frames in each video.
    vid_id, frames = list(map(list, zip(*vid_frames)))
    num_frames = {
        'video_id': vid_id,
        'num_frames': frames
    }

    # Write number of frames to csv.
    df = pd.DataFrame(num_frames)
    df.to_csv(config.num_frames_path(), index=False)

def extract_rois(config):
    """Extracts region of interest.

    # Arguments:
        config: ConfigInterface object.
    """
    sys.path.append('../python')
    import openem

    def _extract_roi(paths):
        img_path, roi_path = paths
        img = openem.Image()
        status = img.FromFile(img_path)
        if status != openem.kSuccess:
            print("Failed to read image {}".format(img_path))
        else:
            roi = openem.Rectify(img, ((x1, y1), (x2, y2)))
            print("Saving ROI to: {}".format(roi_path))
            roi.ToFile(roi_path)

    # Create directories to store ROIs.
    os.makedirs(config.train_rois_dir(), exist_ok=True)

    # Open find ruler results csv.
    endpoints = pd.read_csv(config.find_ruler_inference_path())

    # Build a map between video ID and list of enum containing image
    # and roi paths.
    lookup = {}
    for img_path in config.train_imgs():
        path, f = os.path.split(img_path)
        vid_id = os.path.basename(path)
        roi_dir = os.path.join(config.train_rois_dir(), vid_id)
        os.makedirs(roi_dir, exist_ok=True)
        roi_path = os.path.join(roi_dir, f)
        if vid_id not in lookup:
            lookup[vid_id] = []
        lookup[vid_id].append((img_path, roi_path))


    bar = progressbar.ProgressBar(max_value=len(endpoints),
                                  redirect_stdout=True,
                                  redirect_stderr=True)
    # Extract ROIs.
    pool = ThreadPool(24) # Can't pickle _extract_roi so have to use ThreadPool
    for _, row in bar(endpoints.iterrows()):
        vid_id = row['video_id']
        x1 = row['x1']
        y1 = row['y1']
        x2 = row['x2']
        y2 = row['y2']
        pool.map_async(_extract_roi, lookup[vid_id])


def extract_dets(config):
    """Extracts detection images.

    # Arguments:
        config: ConfigInterface object.
    """
    sys.path.append('../python')
    import openem

    # Create directories to store detections.
    os.makedirs(config.train_dets_dir(), exist_ok=True)

    # Open the detection results csv.
    det_results = pd.read_csv(config.detect_inference_path())

    bar = progressbar.ProgressBar(max_value=len(det_results),
                                  redirect_stdout=True,
                                  redirect_stderr=True)
    # Create the detection images.
    for _, row in bar(det_results.iterrows()):

        # Get the new path.
        vid_id = row['video_id']
        f = "{:04d}.jpg".format(row['frame'])
        roi_path = os.path.join(config.train_rois_dir(), vid_id, f)
        det_dir = os.path.join(config.train_dets_dir(), vid_id)
        os.makedirs(det_dir, exist_ok=True)
        det_path = os.path.join(det_dir, f)

        # Extract detections.
        print("Saving detection image to: {}".format(det_path))
        roi = openem.Image()
        roi.FromFile(roi_path)
        rect = openem.Rect([row['x'], row['y'], row['w'], row['h']])
        det = openem.GetDetImage(roi, rect)
        det.ToFile(det_path)
