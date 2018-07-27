#!/usr/bin/env python

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
    scores = openem.VectorVectorFloat()
    status = classifier.Process(scores)
    if not status == openem.kSuccess:
        raise RuntimeError("Failed to process images!")

    # Display the images and print scores to console.
    for img, s in zip(imgs, scores):
        print("*******************************************")
        print("Fish cover scores:")
        print("No fish:        {}".format(s[0]))
        print("Hand over fish: {}".format(s[1]))
        print("Fish clear:     {}".format(s[2]))
        print("*******************************************")
        print("Fish species scores:")
        print("Fourspot:   {}".format(s[3]))
        print("Grey sole:  {}".format(s[4]))
        print("Other:      {}".format(s[5]))
        print("Plaice:     {}".format(s[6]))
        print("Summer:     {}".format(s[7]))
        print("Windowpane: {}".format(s[8]))
        print("Winter:     {}".format(s[9]))
        print("")
        img.Show()

