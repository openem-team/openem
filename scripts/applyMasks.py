#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import os
import progressbar
import cv2

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Apply image masks prior to training, drop out masked localizations')
    parser.add_argument('population_csv', help='Population csv')
    parser.add_argument('--mask-dir', default='masks')
    parser.add_argument('--output-dir', default='masked_images')
    parser.add_argument('--input-dir', default='images')
    parser.add_argument('--masked-file', default='masked.csv')
    parser.add_argument('--oob-file', default='oob.csv', help='Out of bounds annotations')
    parser.add_argument('--mask-threshold', default=0.50, type=float, help='Percentage of required non-masked pixels to keep annotation')
    
    args = parser.parse_args()
    mask_files={'camera_1': 'camera_1_mask.png',
                'camera_2': 'camera_2_mask.png',
                'camera_3': 'camera_3_mask.png',
                'camera_4': 'camera_4_mask.png'}

    # Load the masks into memory
    masks={}
    for name in mask_files.keys():
        mask_fp=os.path.join(args.mask_dir, mask_files[name])
        mask_data = cv2.imread(mask_fp)
        masks[name] = mask_data
    

    cols=['img_path', 'x1','y1','x2','y2','species']
    population = pd.read_csv(args.population_csv, header=None, names=cols)
    masked_localizations = pd.DataFrame(columns=cols)
    oob_localizations = pd.DataFrame(columns=cols)
    count = len(population)
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=count)
    for idx,row in bar(population.iterrows()):
        image_name = os.path.basename(row.img_path)
        section_dir = os.path.dirname(row.img_path)
        output_dir = os.path.join(args.output_dir, section_dir)
        masked_file = os.path.join(output_dir, image_name)
        os.makedirs(output_dir, exist_ok=True)
        original_image_fp = os.path.join(args.input_dir, row.img_path)
        original_image = cv2.imread(original_image_fp)
        match = False
        for mask_name in masks.keys():
            if mask_name in image_name:
                match = True
                if os.path.exists(masked_file):
                    masked_image = cv2.imread(masked_file)
                else:
                    masked_image = original_image * masks[mask_name]
                    cv2.imwrite(masked_file, masked_image)
                
                # Check to see if localization passes masked logic (>50% unmasked)
                localization_crop = masked_image[row.y1:row.y2, row.x1:row.x2]
                total_pixels = localization_crop.shape[0] * localization_crop.shape[1]
                if total_pixels == 0:
                    print("Zero-sized localization crop")
                    print(row)
                    continue
                # Sets a pixel location to true if it is greater than black (mask value)
                non_black_mask = np.any(localization_crop > np.array([0,0,0]),axis=2)

                # Counts the number of non black pixels
                non_black_pixels = np.count_nonzero(non_black_mask)
                if non_black_pixels / total_pixels > args.mask_threshold:
                    masked_localizations=masked_localizations.append(row)
                else:
                    oob_localizations=oob_localizations.append(row)
                    print(f"Occluded localizations of {row.species} in {row.img_path} coords:{row.x1},{row.y1} to {row.x2},{row.y2}")

            
    masked_localizations.to_csv(args.masked_file, header=False, index=False)
    oob_localizations.to_csv(args.oob_file, header=False, index=False)
    
