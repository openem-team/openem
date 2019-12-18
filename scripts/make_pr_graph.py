#!/usr/bin/env python3

""" Makes a PR curve based on output from generate_detection_metrics.py """

import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_file')
    parser.add_argument('--output', default="pr.png")
    parser.add_argument("--doubles", action="store_true")
    args = parser.parse_args()

    matrix = np.load(args.data_file)
    keep_t = matrix[:,0]
    precision = matrix[:,1]
    recall = matrix[:,2]
    doubles = matrix[:,3]

    fig,ax1 = plt.subplots()
    ax1.plot(recall, precision, color='blue', label='Precision/Recall')
    ax1.set_xlabel('Recall')
    ax1.plot(recall, keep_t, color='green', label='Keep Threshold')

    
    if not args.doubles:
        ax2 = ax1.twinx()
        ax2.plot(recall, doubles, color='red', label='doubles')
        ax2.set_ylabel('doubles', color='red')
        ax2.set_yscale('log')
        ax2.tick_params(axis='y', labelcolor='red')

    fig.legend(loc='lower left')
    fig.tight_layout()
    plt.savefig(args.output)
    
