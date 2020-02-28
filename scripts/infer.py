#!/usr/bin/env python3

""" Run retinanet inference model on a set of images

This is both an example script and one that can be used for network
evaluaton purposes after training.

The script takes a work file, which can be in the openem truth format. The
outputted csv file can be used with this input with `detection_metrics.py` to
acquire precision/recall data.

The inference routine uses a frozen pb retinanet graph.

The input csv work file can be in a couple of flavors.

- The retinanet format used in the training and validation csv formats for that product.
- The openem format used by openem for extracted imagery (vid_id, frame) across a disk layout
  of <vid_id>/<frame:04d>.<img-ext>.
- The video format which can be any CSV file where the first column is a path to a video file

"""

import argparse
from multiprocessing import Process,Queue
import pandas as pd
from openem.Detect import Detection, RetinaNet
from tqdm import tqdm
import cv2
import os
import importlib
import numpy as np
import shutil
import copy
import time

def process_batch_result(args, image_count, result_queue):
    before = time.time()
    results = retinanet.process(batch_size=image_count)
    after = time.time()
    duration = after-before
    print(f"gpu duration = {duration*1000}ms; {1/duration}Hz")
    try:
        result_queue.put_nowait(results)
    except:
        print("Result Queue Stuffed")
        result_queue.put(results)

def process_retinanet_result(args, network_results):
    raw_results=network_results[0]
    sizes=[]
    batch_info=[]
    for cookie in network_results[1]:
        sizes.append(cookie["size"])
        batch_info.append(cookie["batch_info"])

    results=retinanet.format_results(raw_results,
                                     sizes,
                                     threshold=args.keep_threshold)
    new_df = pd.DataFrame(columns=result_cols)
    for batch_idx,batch_result in enumerate(results):
        for result in batch_result:
            confidence_array=np.array(result.confidence)
            confidence = np.max(confidence_array)
            if confidence < args.keep_threshold:
                continue

            conf_as_string = confidence_array.astype(np.str)
            confidence_formatted = ':'.join(list(conf_as_string))
            new_record = {'video_id': batch_info[batch_idx][0],
                          'frame': batch_info[batch_idx][1],
                          'x': result.location[0],
                          'y': result.location[1],
                          'w': result.location[2],
                          'h': result.location[3],
                          'det_species': result.species,
                          'det_conf': confidence_formatted}
            new_df = new_df.append(pd.DataFrame(columns=result_cols,
                                  data=[new_record]))
    new_df.to_csv(args.output_csv, mode='a', header=False,index=False)

def process_video(video_path, retinanet, preprocess_funcs, queue):
    video_reader = cv2.VideoCapture(video_path)
    vid_len = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    count = vid_len
    frame_num = 0
    ok = True
    current_frames=[]
    while ok:
        ok, image_data = video_reader.read()
        if ok:
            processed_image = process_image_data(image_data,
                                                 preprocess_funcs)
            cookie = {"batch_info": (video_id, frame_num)}
            retinanet.addImage(processed_image, cookie)
            frame_num += 1
            current_frames.append(frame_num)
            if len(current_frames) == args.batch_size:
                queue.put(len(current_frames))
                current_frames=[]

    if len(current_frames) > 0:
        queue.put(len(current_frames))
    queue.put(None)

# Static variables for recurrent process_image_data function
def process_image_data(image_data, preprocess_funcs):
    global batch_info
    global image_cnt

    for process in preprocess_funcs:
        image_data = process(video_id, image_data)

    return image_data

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-pb", required=True)
    parser.add_argument("--output-csv", default="results.csv")
    parser.add_argument("--keep-threshold", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--csv-flavor", required=True,
                        help="See format description in top-level help",
                        choices=["retinanet", "openem", "video"])
    parser.add_argument("--img-base-dir", required=True)
    parser.add_argument("--img-ext", default="jpg")
    parser.add_argument("--img-min-side", required=True, type=int)
    parser.add_argument("--img-max-side", required=True, type=int)
    parser.add_argument("--preprocess-module",
                        nargs="+",
                        help="Module name that contains preprocessing function(s) to call on the image prior to insertion into the network")
    parser.add_argument("work_csv", help="CSV with file per row")
    args = parser.parse_args()

    if args.csv_flavor == "retinanet":
        # We only care about the first column
        work_df = pd.read_csv(args.work_csv,
                              header=None)
    elif args.csv_flavor == "openem":
        openem_df = pd.read_csv(args.work_csv)
        media_list = []
        for idx, row in openem_df.iterrows():
            media_list.append(f"{row.video_id}/{row.frame:04d}.{args.img_ext}")
        work_df = pd.DataFrame(data=media_list)
    elif args.csv_flavor == "video":
        video_df = pd.read_csv(args.work_csv, names=None)
        count = len(video_df)
        media_list=list(video_df.iloc[:,0])
        work_df = pd.DataFrame(data=media_list)


    count = len(work_df)

    image_dims = (args.img_min_side, args.img_max_side, 3)
    retinanet = RetinaNet.RetinaNetDetector(args.graph_pb,
                                            imageShape=image_dims,
                                            batch_size=args.batch_size)


    preprocess_funcs=[]
    if args.preprocess_module:
        for module_name in args.preprocess_module:
            module=importlib.import_module(module_name)
            all_funcs=[name for name, f in module.__dict__.items() if callable(f)]
            for name in all_funcs:
                print("Checking {name}")
                if name.startswith('preprocess_'):
                    print(f"Adding preprocessing routine {module}.{name}")
                    preprocess_funcs.append(getattr(module, name))

    image_cnt = 0
    # OpenEM result columns
    result_cols=['video_id', 'frame', 'x','y','w','h', 'det_conf', 'det_species']
    results_df=pd.DataFrame(columns=result_cols)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Outputing results to {args.output_csv}")
    for idx, image in tqdm(enumerate(work_df[0].unique()), desc='Files'):
        image_path = os.path.join(args.img_base_dir, image)
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
            frame = int(os.path.splitext(os.path.basename(image))[0])
        elif args.csv_flavor == 'video':
            video_id = os.path.splitext(os.path.basename(image_path))[0]

        # Now that we have video_id and frame, we can process them
        if args.csv_flavor == "video":
            video_path="/tmp/video.mp4"
            shutil.copyfile(image_path, video_path)
            video_reader = cv2.VideoCapture(video_path)
            vid_len = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            del video_reader

            queue=Queue(maxsize=2)
            result_queue=Queue(maxsize=2)
            def image_consumer(q, result_q):
                with tqdm(total=vid_len, desc="Frames", leave=True) as bar:
                    batch_result = q.get()

                    while batch_result is not None:
                        process_batch_result(args, batch_result, result_q)
                        bar.update(batch_result)
                        try:
                            batch_result = q.get_nowait()
                        except:
                            print("GPU is starved")
                            batch_result = q.get()
                    result_q.put(None)

            def result_consumer(result_q):
                result = result_q.get()

                while result is not None:
                    process_retinanet_result(args, result)
                    try:
                        result = result_q.get_nowait()
                    except:
                        print("Reporter is starved")
                        result = result_q.get()


            reader_thread=Process(target=process_video, args=(video_path, retinanet, preprocess_funcs, queue))
            results_thread=Process(target=result_consumer, args=(result_queue,))
            reader_thread.start()
            results_thread.start()

            # Run the Tensorflow stuff in the main thread
            image_consumer(queue, result_queue)

            reader_thread.join()
            results_thread.join()
        else:
            image_data = cv2.imread(image_path)
            process_image_data(args,
                               preprocess_funcs,
                               video_id,
                               frame,
                               image_data)
