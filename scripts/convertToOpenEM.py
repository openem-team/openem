#!/usr/bin/env python3

""" Convert retinanet data into openem test format """

import argparse
import pandas as pd
import progressbar
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("retinanet_input")
    parser.add_argument("openem_output")
    parser.add_argument("--species-csv", required=True)
    args = parser.parse_args()
    openem_cols=['video_id','frame','x','y','width','height','theta','species_id']
    retinanet_cols = ['img', 'x1','y1','x2','y2', 'species_name']
    retinanet_df = pd.read_csv(args.retinanet_input, header=None, names=retinanet_cols)

    
    
    openem_df = pd.DataFrame(columns=openem_cols)
    openem_df.to_csv(args.openem_output, index=False)
    count = len(retinanet_df)
    bar = progressbar.ProgressBar(max_value=count, redirect_stdout=True)

    species_df = pd.read_csv(args.species_csv, header=None, names=['species','num'])
    
    for idx,row in bar(retinanet_df.iterrows()):
        video_fname = os.path.basename(row.img)
        mp4_pos = video_fname.find('.mp4')
        video_id = video_fname[:mp4_pos]
        frame_with_ext = video_fname[mp4_pos+5:]
        frame = int(os.path.splitext(frame_with_ext)[0])
        species_row = species_df.loc[species_df.species == row.species_name]
        species_id_0 = species_row.iloc[0].num
        species_id_1 = species_id_0 + 1
        datum = {"video_id": video_id,
                 "frame": frame,
                 # Retinanet diagonals can be backwards
                 "x": min(row.x1,row.x2),
                 "y": min(row.y1,row.y2),
                 "width": abs(row.x2-row.x1),
                 "height": abs(row.y2-row.y1),
                 "theta": 0.0,
                 # OpenEM uses 1-based indexing on species
                 'species_id': species_id_1}
        
        datum_df = pd.DataFrame(data=[datum], columns=openem_cols)
        datum_df.to_csv(args.openem_output, header=False, index=False, mode='a')

        
