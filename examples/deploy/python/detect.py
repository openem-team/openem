#!/usr/bin/env python

import argparse
import sys
sys.path.append("../../../python")
sys.path.append("../../python")
import openem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find ruler example.")
    parser.add_argument("pb_file",
        help="Path to protobuf file containing model.",
        type=str)
    parser.add_argument("image_files",
        help="Paths to one or more image files.",
        nargs="+",
        type=str)
    args = parser.parse_args()

    # Create and initialize detector.
    detector = openem.Detector()
    status = detector.Init(args.pb_file)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize detector!")

    # Load in images.
    imgs = [openem.Image() for _ in args.image_files]
    w, h = detector.ImageSize()
    for img, p in zip(imgs, args.image_files):
        status = img.FromFile(p)
        if not status == openem.kSuccess:
            raise IOError("Failed to load image {}".format(p))
        img.Resize(w, h)

    # Add images to processing queue.
    for img in imgs:
        status = detector.AddImage(img)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to add image for processing!")

    # Process the loaded images.
    detections = openem.VectorVectorRect()
    status = detector.Process(detections)
    if not status == openem.kSuccess:
        raise RuntimeError("Failed to process images!")

    # Display the detections on the image.
    for dets, img in zip(detections, imgs):
        for det in dets:
            img.DrawRect(det)
        img.Show()

