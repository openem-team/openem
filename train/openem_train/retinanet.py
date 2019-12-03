""" Wrapper to invoke retinanet training scripts from openem """

import subprocess
import os
import csv

def train(config):
    work_dir = config.work_dir()
    species_csv = os.path.join(train_dir, "retinanet", "species.csv")
    boxes_csv = os.path.join(train_dir, "", "annotations.csv")
    if not os.path.exists(species_csv):
        print(f"Need to make species.csv in {work_dir}")
        print("Attempting to generate it for you from config.ini")
        with open(species_csv,'w') as csv_file:
            writer = csv.writer(csv_file)
            for species in config.species():
                print(f"\t+Adding {species}")
                writer.writerow([species])
            print("Done!")
    else:
        print("Detected Species.csv in training dir")

    args = ['python',
            '/keras_retinanet/scripts/train.py',
            '--train_img_dir',
            config.train_imgs_dir(),
            'openem',
            boxes_csv,
            species_csv]
    p=subprocess.Popen(args)
    p.wait()
    return p.returncode
        
