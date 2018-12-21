"""Functions for preprocessing training data.
"""

import os
import pandas
import scipy.misc
import skimage
from cv2 import VideoCapture
from cv2 import imwrite
from openem_train.util.roi_transform import RoiTransform

def _find_cover_frames(config):
    """ Find frames in cover.csv.

    # Arguments
        config: ConfigInterface object.

    # Returns
        Dict containing video ID as keys, list of frame numbers as values.
    """
    cover = pandas.read_csv(config.cover_path())
    cover_frames = {}
    for _, row in cover.iterrows():
        if row.video_id not in cover_frames:
            cover_frames[row.video_id] = []
        cover_frames[row.video_id].append(int(row.frame))
    return cover_frames

def extract_images(config):
    """Extracts images from video.

    # Arguments
        config: ConfigInterface object.
    """

    # Create directories to store images.
    os.makedirs(config.train_imgs_dir(), exist_ok=True)

    # Read in length annotations.
    ann = pandas.read_csv(config.length_path())
    vid_ids = ann.video_id.tolist()
    ann_frames = ann.frame.tolist()

    # Find frames in cover list.
    cover_frames = _find_cover_frames(config)

    # Start converting images.
    for vid in config.train_vids():
        vid_id, _ = os.path.splitext(os.path.basename(vid))
        img_dir = os.path.join(config.train_imgs_dir(), vid_id)
        os.makedirs(img_dir, exist_ok=True)
        reader = VideoCapture(vid)
        keyframes = [a for a, b in zip(ann_frames, vid_ids) if b == vid_id]
        if vid_id in cover_frames:
            keyframes += cover_frames[vid_id]
        frame = 0
        while reader.isOpened():
            ret, img = reader.read()
            if frame in keyframes:
                img_path = os.path.join(img_dir, '{:04}.jpg'.format(frame))
                print("Saving image to: {}".format(img_path))
                imwrite(img_path, img)
            frame += 1
            if not ret:
                break

def extract_rois(config):
    """Extracts region of interest.

    # Arguments:
        config: ConfigInterface object.
    """

    # Create directories to store ROIs.
    os.makedirs(config.train_rois_dir(), exist_ok=True)

    # Create a transform object.
    roi_transform = RoiTransform(config)

    # Build a map between video ID and list of enum containing image
    # and roi paths.
    lookup = {}
    for img_path in config.train_imgs():
        path, f = os.path.split(img_path)
        vid_id = os.path.basename(path)
        roi_path = os.path.join(config.train_rois_dir(), vid_id, f)
        if vid_id not in lookup:
            lookup[vid_id] = []
        lookup[vid_id].append((img_path, roi_path))

    # Create the ROIs.
    for vid_id in lookup:
        vid_dir = os.path.join(config.train_rois_dir(), vid_id)
        os.makedirs(vid_dir, exist_ok=True)
        tform = roi_transform.transform_for_clip(
            vid_id,
            dst_w=config.detect_width(),
            dst_h=config.detect_height())
        for img_path, roi_path in lookup[vid_id]:
            img = scipy.misc.imread(img_path)
            roi = skimage.transform.warp(
                img,
                tform,
                mode='edge',
                order=3,
                output_shape=(
                    config.detect_height(),
                    config.detect_width()))
            print("Saving ROI to: {}".format(roi_path))
            scipy.misc.imsave(roi_path, roi)

def _get_det_image(row):
    """Extracts detection image using info contained in detection
       inference csv.

    # Arguments
        row: Row of detection inference csv.
    """
    roi = scipy.misc.imread(row['roi_path'])
    x = row['x']
    y = row['y']
    w = row['w']
    h = row['h']
    diff = w - h
    y -= int(diff / 2)
    h = w
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + w > roi.shape[1]:
        w = roi.shape[1] - x
    if y + h > roi.shape[0]:
        h = roi.shape[0] - y
    return roi[y:(y + h), x:(x + w)]

def extract_dets(config):
    """Extracts detection images.

    # Arguments:
        config: ConfigInterface object.
    """

    # Create directories to store detections.
    os.makedirs(config.train_dets_dir(), exist_ok=True)

    # Open the detection results csv.
    det_results = pandas.read_csv(config.detect_inference_path())

    # Create the detection images.
    for _, row in det_results.iterrows():

        # Get the new path.
        path, f = os.path.split(row['roi_path'])
        vid_id = os.path.basename(path)
        det_dir = os.path.join(config.train_dets_dir(), vid_id)
        os.makedirs(det_dir, exist_ok=True)
        det_path = os.path.join(det_dir, f)

        # Extract detections.
        print("Saving detection image to: {}".format(det_path))
        det = _get_det_image(row)
        scipy.misc.imsave(det_path, det)
