import os
import pandas
import glob
import cv2

def extract_images(config):

    # Create directories to store images.
    os.makedirs(config.train_imgs_dir(), exist_ok=True)

    # Read in training data annotations.
    ann = pandas.read_csv(config.train_ann_path())
    vid_ids = ann.video_id.tolist()
    ann_frames = ann.frame.tolist()

    # Start converting images.
    for vid in config.train_vids():
        base, _ = os.path.splitext(os.path.basename(vid))
        img_dir = os.path.join(config.train_imgs_dir(), base)
        os.makedirs(img_dir, exist_ok=True)
        reader = cv2.VideoCapture(vid)
        keyframes = [a for a, b in zip(ann_frames, vid_ids) if b == base]
        frame = 1
        while reader.isOpened():
            ret, img = reader.read()
            if frame in keyframes:
                img_path = os.path.join(img_dir, '{:04}.jpg'.format(frame))
                print("Saving image to: {}".format(img_path))
                cv2.imwrite(img_path, img)
            frame += 1
            if not ret:
                break

