#!/usr/bin/env python

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
    masks = openem.VectorImage()
    status = mask_finder.Process(masks)
    if not status == openem.kSuccess:
        raise RuntimeError("Failed to process images!")

    for mask, img in zip(masks, imgs):
        # Resize the masks back into the same size as the images.
        mask.Resize(img.Width(), img.Height())

        # Check if ruler is present.
        present = openem.RulerPresent(mask)
        if not present:
            print("Could not find ruler in image!  Skipping...")
            continue

        # Find orientation and region of interest based on the mask.
        endpoints = openem.RulerEndpoints(mask)
        r_mask = openem.Rectify(mask, endpoints)
        roi = openem.FindRoi(r_mask)

        # Rectify, crop, and display the image.
        r_img = openem.Rectify(img, transform)
        c_img = openem.Crop(r_img, roi)
        c_img.Show()

