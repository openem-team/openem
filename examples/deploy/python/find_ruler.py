#!/usr/bin/env python

import argparse
import sys
sys.path.append("../../../modules")
import openem
import numpy as np
import matplotlib.pyplot as plt

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

    # Create and initialize the mask finder.
    mask_finder = openem.RulerMaskFinder()
    status = mask_finder.Init(args.pb_file)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize ruler mask finder!")

    # Load in images.
    imgs = [openem.Image() for _ in args.image_files]
    for img, p in zip(imgs, args.image_files):
        status = img.FromFile(p)
        if not status == openem.kSuccess:
            raise IOError("Failed to load image {}".format(p))

    # Add images to processing queue.
    for img in imgs:
        status = mask_finder.AddImage(img)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to add image for processing!")

    # Process the loaded images.
    masks = openem.vector_image()
    status = mask_finder.Process(masks)
    if not status == openem.kSuccess:
        raise RuntimeError("Failed to process images!")

    for i in range(len(masks)):
        # Resize the masks back into the same size as the images.
        masks[i].Resize(imgs[i].Width(), imgs[i].Height())

        # Check if ruler is present.
        present = openem.RulerPresent(masks[i])
        if not present:
            print("Could not find ruler in image!  Skipping...")
            continue

        # Find orientation and region of interest based on the mask.
        transform = openem.RulerOrientation(masks[i])
        r_mask = openem.Rectify(masks[i], transform)
        roi = openem.FindRoi(r_mask)

        # Rectify, crop, and display the image.
        r_img = openem.Rectify(imgs[i], transform)
        c_img = openem.Crop(r_img, roi)
        w = c_img.Width()
        h = c_img.Height()
        ch = c_img.Channels()
        disp_img = c_img.DataCopy()
        disp_img = np.array(disp_img)
        disp_img = np.reshape(disp_img, (h, w, ch))
        disp_img = np.flip(disp_img, axis=2)
        plt.imshow(disp_img)
        plt.show()

