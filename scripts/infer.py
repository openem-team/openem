#!/usr/bin/env python3

""" Run retinanet inference model on a set of images """

import argparse
import pandas as pd
from openem.Detect import Detection, RetinaNet
import progressbar
import cv2
import os

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-pb", required=True)
    parser.add_argument("--output-csv", default="results.csv")
    parser.add_argument("--keep-threshold", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--csv-flavor", required=True,
                        choices=["retinanet", "openem"])
    parser.add_argument("--img-base-dir", required=True)
    parser.add_argument("--img-ext", default="jpg")
    parser.add_argument("--img-min-side", required=True, type=int)
    parser.add_argument("--img-max-side", required=True, type=int)
    parser.add_argument("work_csv", help="CSV with file per row")
    args = parser.parse_args()

    if args.csv_flavor == "retinanet":
        # We only care about the first column
        work_df = pd.read_csv(args.work_csv,
                              header=None)
    elif args.csv_flavor == "openem":
        openem_df = pd.read_csv(args.work_csv)
        media_list = []
        for idx, openem_row in openem_df.iterrows():
            media_list.append(f"{row.vid_id}/{row.frame}.{args.img_ext}")
        work_df = pd.DataFrame(data=media_list)

    count = len(work_df)
    bar = progressbar.ProgressBar(redirect_stdout=True, max_value=count)

    image_dims = (args.img_min_side, args.img_max_side)
    retinanet = RetinaNet.RetinaNetDetector(args.graph_pb,
                                            imageShape=image_dims)

    image_cnt = 0
    # OpenEM result columns
    result_cols=['video_id', 'frame', 'x','y','w','h', 'det_conf', 'det_species']
    results_df=pd.DataFrame(columns=result_cols)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Outputing results to {args.output_csv}")
    for idx, image in bar(work_df.iterrows()):
        image_path = os.path.join(args.img_base_dir, image[0])
        image_data = cv2.imread(image_path)
        if args.csv_flavor == 'retinanet':
            # Raw video inputs may look like this:
            # <section>/4996995_camera_1_2019_07_06-11_10.mp4_290.png
            video_fname = os.path.basename(image_path)
            mp4_pos = video_fname.find('.mp4')
            video_id = video_fname[:mp4_pos]
            frame_with_ext = video_fname[mp4_pos+5:]
            frame = int(os.path.splitext(frame_with_ext)[0])
        elif args.csv_flavor == 'openem':
            video_id = os.path.basename(os.path.dirname(image_path))
            frame = int(os.path.splitext(image)[0])

        retinanet.addImage(image_data)
        image_cnt += 1
        if image_cnt == args.batch_size or idx + 1 == count:
            image_cnt = 0
            results = retinanet.process()
            for batch_result in results:
                for result in batch_result:
                    new_record = {'video_id': video_id,
                                  'frame': frame,
                                  'x': result.location[0],
                                  'y': result.location[1],
                                  'w': result.location[2],
                                  'h': result.location[3],
                                  'det_species': result.species,
                                  'det_conf': result.confidence}
                    new_df = pd.DataFrame(columns=result_cols,
                                          data=[new_record])
                    new_df.to_csv(args.output_csv, mode='a', header=False,index=False)
