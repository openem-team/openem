import os
import pandas
import glob
import cv2

def extract_images(config):

    # Get paths from config.
    work_dir = config.get('Paths', 'WorkDir')
    train_dir = config.get('Paths', 'TrainDir')

    # Create directories to store images.
    train_imgs_dir = os.path.join(work_dir, 'train_imgs')
    os.makedirs(train_imgs_dir, exist_ok=True)

    # Get list of all training videos.
    train_vids_patt = os.path.join(train_dir, 'train_videos', '*.mp4')
    train_vids = glob.glob(train_vids_patt)

    # Read in training data annotations.
    train_ann_path = os.path.join(train_dir, 'train_annotations.csv')
    ann = pandas.read_csv(train_ann_path)
    vid_ids = ann.video_id.tolist()
    ann_frames = ann.frame.tolist()

    # Start converting images.
    for vid in train_vids:
        base, _ = os.path.splitext(os.path.basename(vid))
        img_dir = os.path.join(train_imgs_dir, base)
        os.makedirs(img_dir, exist_ok=True)
        reader = cv2.VideoCapture(vid)
        keyframes = [a for a, b in zip(ann_frames, vid_ids) if b == base]
        frame = 0
        while reader.isOpened():
            ret, img = reader.read()
            if frame in keyframes:
                img_path = os.path.join(img_dir, '{:04}.jpg'.format(frame))
                cv2.imwrite(img_path, img)
            frame += 1

