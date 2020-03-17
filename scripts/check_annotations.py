#!/usr/bin/env python3

'''
    This script takes in an input csv file and removes any annotations that have y1 >= y2
    or x1 >= x2.  It will write to a csv file if one is provided.  
'''

import os
import sys
import pandas as pd
import numpy as np

def read_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", help="Input CSV file.")
    parser.add_argument("-o", "--output", help="Output CSV file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = read_args()

    annotations = pd.read_csv(args.input, header=0)

    indices = []
    for idx, row in annotations.iterrows():
        if row[2] >= row[4]:
            indices.append(idx)
        elif row[1] >= row[3]:
            indices.append(idx)
    print(f"Removing indices: {indices}")
    
    updated_annotations = annotations.drop(indices)

    if args.output:
        updated_annotations.to_csv(args.output, index=False)