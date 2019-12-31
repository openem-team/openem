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

    fig,axes = plt.subplots(2)
    ax1 = axes[0]
    ax1.plot(recall, precision, color='blue', label='Precision/Recall')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    #ax1.plot(recall, keep_t, color='green', label='Keep Threshold')
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1)
    ax1.grid()
    ax2 = axes[1]
    ax2.plot(recall, keep_t,color='green', label='Keep Threshold')
    ax2.set_ylabel('Keep Threshold')
    ax2.set_xlabel('Recall')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.grid()
    if not args.doubles:
        ax2 = ax1.twinx()
        ax2.plot(recall, doubles, color='red', label='doubles')
        ax2.set_ylabel('Doubles / Total Truth', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

    fig.legend(loc='lower left')
    fig.tight_layout()
    plt.savefig(args.output)
    
