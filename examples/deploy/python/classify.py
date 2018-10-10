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
    parser = argparse.ArgumentParser(description="Classify example.")
    parser.add_argument("pb_file",
        help="Path to protobuf file containing model.",
        type=str)
    parser.add_argument("image_files",
        help="Paths to one or more image files.", 
        nargs="+",
        type=str)
    args = parser.parse_args()

    # Create and initialize classifier.
    classifier = openem.Classifier()
    status = classifier.Init(args.pb_file)
    if not status == openem.kSuccess:
        raise IOError("Failed to initialize classifier!")

    # Load in images.
    imgs = [openem.Image() for _ in args.image_files]
    w, h = classifier.ImageSize()
    for img, p in zip(imgs, args.image_files):
        status = img.FromFile(p)
        if not status == openem.kSuccess:
            raise IOError("Failed to load image {}".format(p))
        img.Resize(w, h)

    # Add images to processing queue.
    for img in imgs:
        status = classifier.AddImage(img)
        if not status == openem.kSuccess:
            raise RuntimeError("Failed to add image for processing!")

    # Process the loaded images.
    scores = openem.VectorClassification()
    status = classifier.Process(scores)
    if not status == openem.kSuccess:
        raise RuntimeError("Failed to process images!")

    # Display the images and print scores to console.
    for img, s in zip(imgs, scores):
        print("*******************************************")
        print("Fish cover scores:")
        print("No fish:        {}".format(s.cover[0]))
        print("Hand over fish: {}".format(s.cover[1]))
        print("Fish clear:     {}".format(s.cover[2]))
        print("*******************************************")
        print("Fish species scores:")
        print("Fourspot:   {}".format(s.species[0]))
        print("Grey sole:  {}".format(s.species[1]))
        print("Other:      {}".format(s.species[2]))
        print("Plaice:     {}".format(s.species[3]))
        print("Summer:     {}".format(s.species[4]))
        print("Windowpane: {}".format(s.species[5]))
        print("Winter:     {}".format(s.species[6]))
        print("")
        img.Show()

