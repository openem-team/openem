"""Functions for preprocessing training data.
"""

import os
import pandas
from cv2 import VideoCapture
from cv2 import imwrite

def _find_no_fish(config):
    """ Find frames containing no fish.

    # Arguments
        config: ConfigInterface object.

    # Returns
        Dict containing video ID as keys, list of frame numbers as values.
    """
    cover = pandas.read_csv(config.cover_path())
    no_fish = {}
    for _, row in cover.iterrows():
        if row.cover == 0:
            if row.video_id not in no_fish:
                no_fish[row.video_id] = []
            no_fish[row.video_id].append(int(row.frame))
    return no_fish

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

    # Find frames containing no fish.
    no_fish = _find_no_fish(config)

    # Start converting images.
    for vid in config.train_vids():
        vid_id, _ = os.path.splitext(os.path.basename(vid))
        img_dir = os.path.join(config.train_imgs_dir(), vid_id)
        os.makedirs(img_dir, exist_ok=True)
        reader = VideoCapture(vid)
        keyframes = [a for a, b in zip(ann_frames, vid_ids) if b == vid_id]
        if vid_id in no_fish:
            keyframes += no_fish[vid_id]
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
