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
import threading
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
import tator

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
    cookies = network_results[1]
    sizes=[cookie["size"] for cookie in cookies]
    batch_info=[cookie["batch_info"] for cookie in cookies]

    results=retinanet.format_results(raw_results,
                                     sizes,
                                     threshold=args.keep_threshold)
    data = []
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
            data.append(new_record)
    new_df = pd.DataFrame(columns=result_cols,
                          data=data)
    new_df.to_csv(args.output_csv, mode='a', header=False,index=False)

current_frames = 0
frame_lock = threading.Lock()
def process_frame(image, preprocess_funcs, cookie, queue):
    global current_frames
    processed_image = process_image_data(image,
                                         preprocess_funcs)
    retinanet.addImage(processed_image, cookie)
    with frame_lock:
        current_frames += 1
        if current_frames >= args.batch_size:
            try:
                queue.put_nowait(args.batch_size)
            except:
                print("GPU Stuffed")
                queue.put(args.batch_size)
            finally:
                current_frames-=args.batch_size

def process_video(video_q, preprocess_funcs, queue):
    video_tuple = video_q.get()
    while video_tuple != None:
        video_path, video_id = video_tuple
        video_reader = cv2.VideoCapture(video_path)
        vid_len = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        count = vid_len
        frame_num = 0
        threads = []
        ok = True
        while ok:
            ok, image_data = video_reader.read()
            if ok:
                for idx,thread in enumerate(threads):
                    if thread.is_alive() == False:
                        thread.join()
                        del threads[idx]
                cookie = {"batch_info": (video_id, frame_num)}
                if args.batch_size > 2:
                    thread = threading.Thread(target=process_frame,
                                              args=(image_data, preprocess_funcs, cookie, queue))
                    thread.start()
                    threads.append(thread)
                    # Gate if we get a ton o' threads
                    while len(threads) >= args.batch_size*2:
                        for idx,t in enumerate(threads):
                            if t.is_alive() == False:
                                t.join()
                                del threads[idx]
                else:
                    process_frame(image_data, preprocess_funcs, cookie, queue)
                frame_num += 1

        if len(threads) > 0:
            for thread in threads:
                thread.join()
        queue.put(None)
        video_tuple = video_q.get()
    print("Exiting process video")

# Static variables for recurrent process_image_data function
def process_image_data(image_data, preprocess_funcs):
    global batch_info
    global image_cnt

    for process in preprocess_funcs:
        image_data = process(video_id, image_data)

    return image_data

def image_consumer(q, result_q,vid_len):
    with tqdm(total=vid_len, desc="Frames", leave=True) as bar:
        batch_result = q.get()
        print(f"Initial batch size = {batch_result}")
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
    print("starting results")
    result = result_q.get()

    while result is not None:
        process_retinanet_result(args, result)
        try:
            result = result_q.get_nowait()
        except:
            result = result_q.get()
    print("Exiting results")

def result_consumer_v2(name_q,result_q):
    print("starting results")
    name = name_q.get()
    while name is not None:
        print(f"Results for {name}")
        result = result_q.get()
        while result is not None:
            process_retinanet_result(args, result)
            try:
                result = result_q.get_nowait()
            except:
                result = result_q.get()
        name = name_q.get()
    print("Exiting results")

def get_videoId_frame(args, image_path):
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
        frame = None
    elif args.csv_flavor == 'image':
        video_id = os.path.splitext(os.path.basename(image_path))[0]
        frame = 0
    return video_id, frame

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-pb", required=True)
    parser.add_argument("--output-csv", default="results.csv")
    parser.add_argument("--keep-threshold", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--csv-flavor", required=True,
                        help="See format description in top-level help",
                        choices=["retinanet",
                                 "openem",
                                 "video",
                                 "image"])
    parser.add_argument("--img-base-dir", required=True)
    parser.add_argument("--img-ext",
                        default="jpg",
                        help="Only required for OpenEM Flavor")
    parser.add_argument("--img-min-side", required=True, type=int)
    parser.add_argument("--img-max-side", required=True, type=int)
    parser.add_argument("--preprocess-module",
                        nargs="+",
                        help="Module name that contains preprocessing function(s) to call on the image prior to insertion into the network")
    parser.add_argument("--cpu-only",
                        action="store_true")
    parser.add_argument("--host",
                        type=str,
                        help="Tator host to use")
    parser.add_argument("--token",
                        type=str,
                        help="Token to use for tator.")
    parser.add_argument("work_csv", help="CSV with file per row")
    args = parser.parse_args()

    if args.token:
        api = tator.get_api(host=args.host,token=args.token)
    else:
        api = None


    if args.cpu_only == True:
        print("Enabling CPU-only inference")

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
    elif args.csv_flavor == "video" or args.csv_flavor == "image":
        video_df = pd.read_csv(args.work_csv, header=None)
        count = len(video_df)
        media_list=list(video_df.iloc[:,0])
        work_df = pd.DataFrame(data=media_list)


    count = len(work_df)

    image_dims = (args.img_min_side, args.img_max_side,3)
    retinanet = RetinaNet.RetinaNetDetector(args.graph_pb,
                                            imageShape=image_dims,
                                            batch_size=args.batch_size,
                                            cpu_only=args.cpu_only)


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

    batch_queue=Queue(maxsize=4)
    result_queue=Queue(maxsize=2)
    name_queue=Queue(maxsize=2)
    name_res_queue=Queue(maxsize=2)
    print(work_df)
    media_files = work_df[0].unique()
    if args.csv_flavor != "video":
        # We are iterating over images
        def image_reader(b_queue):
            num_images = 0
            for image in tqdm(media_files, desc='Files'):
                image_path = os.path.join(args.img_base_dir, image)
                video_id, frame = get_videoId_frame(args, image_path)
                need_to_delete = False
                if not os.path.exists(image_path) and not api is None:
                    print(f"Downloading {image} from Tator.")
                    media_id = image.split('_')[0]
                    media_element = api.get_media(media_id)
                    for _ in tator.util.download_media(api, media_element, image_path):
                        pass
                    need_to_delete = True
                    assert os.path.exists(image_path)
                elif not os.path.exists(image_path):
                    print(f"{image_path} not found!")
                    print("No Tator Connection info provided.")
                image_data = cv2.imread(image_path)
                if need_to_delete and image_path.startswith('/tmp'):
                    os.remove(image_path)
                cookie = {"batch_info": (video_id, frame)}
                retinanet.addImage(image_data, cookie)
                num_images += 1
                if num_images >= args.batch_size:
                    b_queue.put(args.batch_size)
                    num_images -= args.batch_size
            if num_images > 0:
                b_queue.put(num_images)
                num_images = 0
            b_queue.put(None)

        reader_thread=Process(target=image_reader,
                              args=(batch_queue,))
        results_thread=Process(target=result_consumer,
                               args=(result_queue,))
        reader_thread.start()
        results_thread.start()

        # Run the Tensorflow stuff in the main thread
        image_consumer(batch_queue, result_queue, len(media_files))

        reader_thread.join()
        results_thread.join()

    else:
        reader_thread=Process(target=process_video,
                              args=(name_queue,
                                    preprocess_funcs, batch_queue))
        results_thread=threading.Thread(target=result_consumer_v2,
                               args=(name_res_queue,result_queue,))
        print(f"Made new Process Objects")
        reader_thread.start()
        print("Launched video Reader")
        results_thread.start()
        print("Started all Threads")
        # For each video spawn up a worker thread combo
        for video in tqdm(media_files, desc='Files'):
            need_to_delete = False
            print(f"Processing {video}")
            image_path = os.path.join(args.img_base_dir, video)
            if not os.path.exists(image_path) and not api is None:
                print(f"Downloading {video} from Tator.")
                media_id = video.split('_')[0]
                media_element = api.get_media(media_id)
                for _ in tator.util.download_media(api, media_element, image_path):
                    pass
                need_to_delete = True
                assert os.path.exists(image_path)
            elif not os.path.exists(image_path):
                print(f"{image_path} not found!")
                print("No Tator Connection info provided.")
            video_id, frame = get_videoId_frame(args, image_path)
            # Now that we have video_id and frame, we can process them

            #print(f"Starting Copy")
            #video_path="/tmp/video.mp4"
            #shutil.copyfile(image_path, video_path)
            #print(f"Finished copy")
            video_path=image_path
            video_reader = cv2.VideoCapture(video_path)
            vid_len = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
            del video_reader

            name_queue.put((video_path,video_id))
            name_res_queue.put(video_path)
            # Run the Tensorflow stuff in the main thread
            image_consumer(batch_queue, result_queue, vid_len)
            print("Consumer Exited")

            # Delete only temporary files
            if need_to_delete and video_path.startswith('/tmp'):
                os.remove(video_path)

        name_queue.put(None)
        name_res_queue.put(None)
        reader_thread.join()
        print("Joining reader thread")
        results_thread.join()
        print("Joining results thread")
