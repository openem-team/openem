#!/usr/bin/env python3

""" Makes a PR curve based on output from generate_detection_metrics.py """

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from cycler import cycler

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_file')
    parser.add_argument('--output', default="pr.png")
    parser.add_argument("--doubles", action="store_true")
    parser.add_argument("--species")
    
    args = parser.parse_args()

    species_lookup={}
    if args.species:
        species_df=pd.read_csv(args.species, names=["species", "id_0"], header=None)
        for idx,row in species_df.iterrows():
            species_lookup[row.id_0+1] = row.species

    with open(args.data_file, 'rb') as data:
        bySpecies = pickle.load(data)

    default_cycler = (cycler(color=['r', 'g', 'b', 'y','c','m','y','k']) *
                      cycler(linestyle=['-', '--', ':', '-.']))

    plt.rc('axes', prop_cycle=default_cycler)
    combined=plt.figure()
    combined_ax = combined.subplots(1)
    # Only look at interesting area for combined
    # plot
    combined_ax.set_xlim(0,1)
    combined_ax.set_ylim(0,1)
    combined.set_figheight(10)
    combined.set_figwidth(8)
    for species in bySpecies:
        matrix = np.array(bySpecies[species])
        keep_t = matrix[:,0]
        precision = matrix[:,1]
        recall = matrix[:,2]
        doubles = matrix[:,3]
        if species is None:
            species_name="All"
        else:
            species_name=species_lookup.get(species, species)
        combined_ax.plot(recall, precision, label=species_name)
    combined_ax.legend()
    plt.savefig('pr_combined.png')
    print("Writing pr_combined.png")
    plt.close(combined)
    for species in bySpecies:
        matrix = np.array(bySpecies[species])
        keep_t = matrix[:,0]
        precision = matrix[:,1]
        recall = matrix[:,2]
        doubles = matrix[:,3]
        fig,axes = plt.subplots(2)
        fig.set_figheight(10)
        fig.set_figwidth(8)
        ax1 = axes[0]
        if species:
            ax1.set_title(species)
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
        if species:
            species_name=species_lookup.get(species, species)
            output=f"pr_{species_name}.png"
        else:
            output=args.output
        plt.savefig(output)
        print(f"Writing {output}")
        plt.close(fig)
    
