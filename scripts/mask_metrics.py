#!/usr/bin/env python

import argparse
import os
import glob
import cv2
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description='Computes mask accuracy for a series of integration times.',
    )
    parser.add_argument(
        'inference_dir',
        help='Mask output directory. One subdirectory per video, with files '
        'named by frame number.',
    )
    parser.add_argument(
        'truth_dir',
        help='Mask ground truth. Should contain one mask per video. File names '
        'should match subdirectories in inference_dir.',
    )
    parser.add_argument(
        '--min_frames',
        help='Minimum number of frames to integrate before applying threshold.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--max_frames',
        help='Maximum number of frames to integrate before applying threshold.',
        type=int,
        default=300,
    )
    parser.add_argument(
        '--inc_frames',
        help='Increment between integration.',
        type=int,
        default=5,
    )
    return parser.parse_args()

def integrate(mask_dir, keep_frames):
    img_sum = None
    num_imgs = 0.0
    sums = []
    for frame, fname in enumerate(sorted(os.listdir(mask_dir))):
        # Don't include mask average
        if fname == '_mask_avg.png':
            continue

        # Read in the frame
        img = cv2.imread(os.path.join(mask_dir, fname))
        img = img.astype(np.float64)

        # Add frame to sum
        if img_sum is None:
            img_sum = img
        else:
            img_sum += img
        num_imgs += 1.0

        # If needed, keep this frame
        if frame in keep_frames:
            keep = img_sum / num_imgs
            ret, keep = cv2.threshold(keep, 127, 255, cv2.THRESH_BINARY)
            sums.append(keep)
            
    return sums

def get_truth(mask_dir, truth_dir):
    vid_id = os.path.basename(mask_dir)
    truth_path = glob.glob(os.path.join(truth_dir, '*' + vid_id + '.*'))[0]
    if not os.path.exists(truth_path):
        raise ValueError(f"Couldn't find truth mask {truth_path}!")
    return cv2.imread(truth_path)

def compare(sums, truth):
    accs = []
    truth = truth / 255.0
    for predict in sums:
        predict = predict / 255.0
        xor = np.logical_xor(predict.astype(np.bool), truth.astype(np.bool))
        wrong = np.sum(xor)
        total = xor.shape[0] * xor.shape[1]
        acc = (total - wrong) / total
        accs.append(acc)
    return accs

if __name__ == '__main__':
    args = parse_args()
    accs = [] # accuracy for every video
    keep_frames = np.arange(args.min_frames, args.max_frames, args.inc_frames)
    for mask_dir in os.listdir(args.inference_dir):
        sums = integrate(os.path.join(args.inference_dir, mask_dir), keep_frames)
        truth = get_truth(mask_dir, args.truth_dir)
        acc = compare(sums, truth)
        accs.append(acc)
    accs = np.array(accs)
    np.save('accs.npy', accs)
    np.save('keep.npy', keep_frames)
