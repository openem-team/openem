#!/usr/bin/env python3

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from pprint import pprint

from . import thumbnail_classifier

def main():
    """ Invokes a test of the thumbnail classifier """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", help="Directory containing model images")
    parser.add_argument("--image-size", default="224x224",
                        help="WxH to use for network input size")
    parser.add_argument("--dropout-percentage",
                        default=0.20,
                        type=float,
                        help="Percentage of network to dropout for each MC run")
    parser.add_argument("--monte-carlo-runs", default=20, type=int,
                        help="Number of Monte Carlo runs per network")
    parser.add_argument("--batch-size", default=16,
                        help="Number of thumbnails to process in a batch")
    parser.add_argument("track_dir",
                        help="Directory containing track folders")
    parser.add_argument("output_file",
                        help="Name of output file")
    args = parser.parse_args()
    print(args)

    model_files = [os.path.join(args.model_dir,f) for f in \
                   os.listdir(args.model_dir)]
    

    image_size = [int(x) for x in args.image_size.split('x')]
    classifier = thumbnail_classifier.EnsembleClassifier(
        model_files,
        ["Commercial",
         "Recreational"],
        imageSize=image_size,
        dropoutPercent=args.dropout_percentage,
        monteCarloRuns=args.monte_carlo_runs,
        batchSize=args.batch_size)

    results = []
    for root,dirs,files in os.walk(args.track_dir):
        track_id = os.path.basename(root)
        images=[]
        for fp in files:
            bgr = cv2.imread(os.path.join(root,fp))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = rgb.astype(np.float64)
            rgb /= 255.0
            images.append(rgb)
        if images:
            detection_scores,entropy = classifier.run_track(images)
            label, winner, track_entropy = classifier.process_track_results(
                detection_scores,
                entropy)
            results.append({"track_id": track_id,
                            "label": label,
                            "entropy": track_entropy})

            
            
                                                         
    df = pd.DataFrame(columns=results[0].keys(),
                      data=results)
    df.to_csv(args.output_file, index=False)
    

if __name__=="__main__":
    main()
