#!/usr/bin/env python

import argparse
import sys
sys.path.append("../../../modules")
import openem
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util import image_to_numpy

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
        raise IOError("Failed to initialize ruler mask finder!")

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
    detections = openem.vector_vector_rect()
    status = detector.Process(detections)
    if not status == openem.kSuccess:
        raise RuntimeError("Failed to process images!")

    # Display the detections on the image.
    for dets, img in zip(detections, imgs):
        disp_img = image_to_numpy(img)
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.imshow(disp_img)
        for det in dets:
            x, y, w, h = det
            rect = patches.Rectangle((x, y), w, h, 
                facecolor="none", 
                linewidth=2, 
                edgecolor="r")
            ax.add_patch(rect)
        plt.show()

