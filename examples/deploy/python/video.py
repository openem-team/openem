#!/usr/bin/env python

import argparse
import sys
import math
sys.path.append("../../../python")
sys.path.append("../../python")
import openem

def find_roi(mask_finder_path, vid_path):
    """Finds ROI in a video.

    # Arguments
        mask_finder_path: Path to find_ruler model file.
        vid_path: Path to the video.

    # Returns:
        Region of interest and affine transform.

    # Raises:
        IOError: If video or model file cannot be opened.
        RuntimeError: If fails to add images or process model.
    """
    # Determined by experimentation with GPU having 8GB memory.
    max_img = 8

    # Create and initialize the mask finder.
    mask_finder = openem.RulerMaskFinder()
    status = mask_finder.Init(mask_finder_path)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize mask finder!")

    # Decode the first 100 frames and find the mask that corresponds
    # to the largest ruler area.
    reader = openem.VideoReader()
    status = reader.Init(vid_path)
    if not status == openem.kSuccess:
        raise IOError("Failed to open video!")
    masks = openem.VectorImage()
    best_mask = openem.Image()
    max_mask_sum = 0.0
    vid_end = False
    for i in range(math.ceil(100 / max_img)):
        for j in range(max_img):
            img = openem.Image()
            status = reader.GetFrame(img)
            if not status == openem.kSuccess:
                vid_end = True
                break
            status = mask_finder.AddImage(img)
            if not status == openem.kSuccess:
                raise RuntimeError("Failed to add frame to mask finder!")
        status = mask_finder.Process(masks)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to process mask finder!")
        for mask in masks:
            mask_sum = mask.Sum()[0]
            if mask_sum > max_mask_sum:
                max_mask_sum = mask_sum
                best_mask = mask
        if vid_end:
            break

    # Now that we have the best mask, use this to compute the ROI.
    best_mask.Resize(reader.Width(), reader.Height())
    transform = openem.RulerOrientation(best_mask)
    r_mask = openem.Rectify(best_mask, transform)
    roi = openem.FindRoi(r_mask)
    return (roi, transform)

def detect_and_classify(detect_path, classify_path, vid_path, roi, transform):
    """Finds and classifies detections for all frames in a video.

    # Arguments
        detect_path: Path to detect model file.
        classify_path: Path to classify model file.
        vid_path: Path to the video.
        roi: Region of interest output from find_roi.
        transform: Transform output from find_roi.

    # Returns
        Detection rects and classification scores.

    # Raises
        IOError: If video or model files cannot be opened.
        RuntimeError: If unable to add frame or process a model.
    """
    # Determined by experimentation with GPU having 8GB memory.
    max_img = 32

    # Create and initialize the detector.
    detector = openem.Detector()
    status = detector.Init(detect_path, 0.5)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize detector!")

    # Create and initialize the classifier.
    classifier = openem.Classifier()
    status = classifier.Init(classify_path, 0.5)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize classifier!")

    # Initialize the video reader.
    reader = openem.VideoReader()
    status = reader.Init(vid_path)
    if not status == openem.kSuccess:
        raise IOError("Failed to open video {}!".format(vid_path))

    # Iterate through frames.
    vid_end = False
    detections = []
    scores = []
    while True:

        # Find detections.
        dets = openem.VectorVectorRect()
        imgs = [openem.Image() for _ in range(max_img)]
        for i, img in enumerate(imgs):
            status = reader.GetFrame(img)
            if not status == openem.kSuccess:
                vid_end = True
                break
            img = openem.Rectify(img, transform)
            img = openem.Crop(img, roi)
            status = detector.AddImage(img)
            imgs[i] = img
            if not status == openem.kSuccess:
                raise RuntimeError("Failed to add frame to detector!")
        status = detector.Process(dets)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to process detector!")
        detections += dets

        # Classify detections
        for det_frame, img in zip(dets, imgs):
            score_batch = openem.VectorVectorFloat()
            for det in det_frame:
                det_img = openem.GetDetImage(img, det)
                status = classifier.AddImage(det_img)
                if not status == openem.kSuccess:
                    raise RuntimeError("Failed to add frame to classifier!")
            status = classifier.Process(score_batch)
            if not status == openem.kSuccess:
                raise RuntimeError("Failed to process classifier!")
            scores.append(score_batch)
        if vid_end:
            break
    return (detections, scores)

def write_video(vid_path, out_path, roi, transform, detections, scores):
    """Writes a new video with bounding boxes around detections.

    # Arguments
        vid_path: Path to the original video.
        out_path: Path to the output video.
        roi: Region of interest output from find_roi.
        transform: Transform output from find_roi.
        detections: Detections for each frame.
        scores: Cover and species scores for each detection.
    """
    # Initialize the video reader.
    reader = openem.VideoReader()
    status = reader.Init(vid_path)
    if not status == openem.kSuccess:
        raise IOError("Failed to read video {}!".format(vid_path))

    # Initialize the video writer.
    writer = openem.VideoWriter()
    status = writer.Init(
        out_path,
        reader.FrameRate(),
        openem.kWmv2,
        (reader.Width(), reader.Height()))
    if not status == openem.kSuccess:
        raise IOError("Failed to write video {}!".format(out_path))

    # Iterate through frames.
    for det_frame, score_frame in zip(detections, scores):
        frame = openem.Image()
        status = reader.GetFrame(frame)
        if not status == openem.kSuccess:
            raise RuntimeError("Error retrieving video frame!")
        frame.DrawRect(roi, (255, 0, 0), 1, transform)
        for j, (det, score) in enumerate(zip(det_frame, score_frame)):
            clear = score[2]
            hand = score[1]
            if j == 0:
                if clear > hand:
                    frame.DrawText("Clear", (0, 0), (0, 255, 0))
                    det_color = (0, 255, 0)
                else:
                    frame.DrawText("Hand", (0, 0), (0, 0, 255))
                    det_color = (0, 0, 255)
            frame.DrawRect(det, det_color, 2, transform, roi)
        status = writer.AddFrame(frame)
        if not status == openem.kSuccess:
            raise RuntimeError("Error adding frame to video!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End to end example on a video clip.")
    parser.add_argument("find_ruler_model",
        type=str,
        help="Path to pb file with find_ruler model.")
    parser.add_argument("detect_model",
        type=str,
        help="Path to pb file with detect model.")
    parser.add_argument("classify_model",
        type=str,
        help="Path to pb file with classify model.")
    parser.add_argument("video_paths",
        type=str,
        nargs="+",
        help="One or more paths to video files.")
    args = parser.parse_args()
    for i, video_path in enumerate(args.video_paths):
        # Find the ROI.
        print("Finding region of interest...")
        roi, transform = find_roi(args.find_ruler_model, video_path)

        # Find detections and classify them.
        print("Performing detection and classification...")
        detections, scores = detect_and_classify(
            args.detect_model,
            args.classify_model,
            video_path,
            roi,
            transform)

        # Write annotated video to file.
        print("Writing video to file...")
        write_video(
            video_path,
            "annotated_video_{}.avi".format(i),
            roi,
            transform,
            detections,
            scores)
