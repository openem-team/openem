#!/usr/bin/python3

import argparse
import numpy as np
import socket
import cv2
import time
import yaml
import multiprocessing
from multiprocessing.sharedctypes import RawArray
import ctypes
import signal
import subprocess
import datetime
import requests

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

import torch
import time
import tempfile
import random
import os
import tator
import torchvision
import threading


"""
Example usage:
shell>  chrt -f 50 python3 stream_detect.py broadcast_example.yaml


shell>  python3 stream_detect.py -v file_example.yaml
"""

def get_tator_api(strategy):
  if os.getenv('TATOR_API_SERVICE'):
    host = os.getenv("TATOR_API_SERVICE").replace('/rest','')
    token=os.getenv('TATOR_AUTH_TOKEN')
    return tator.get_api(host, token)
  else:
    try:
      host = strategy['tator']['host']
      token = strategy['tator']['token']
      return tator.get_api(host,token)
    except:
      print("ERROR: No tator credentials provided.")
      return None


def get_tator_project(strategy):
  project_id=os.getenv('TATOR_PROJECT_ID')
  if project_id:
    return project_id
  else:
    return strategy['tator']['project']

def get_best_quality(media_object):
  media_object = media_object.to_dict()
  if media_object['media_files'] == None:
    return None
  archival=media_object['media_files'].get('archival', [])
  streaming=media_object['media_files'].get('streaming', [])
  images=media_object['media_files'].get('image', [])
  if archival:
    archival.sort(key=lambda x:x['resolution'][0], reverse=True)
    return archival[0]['path']
  elif streaming:
    streaming.sort(key=lambda x:x['resolution'][0], reverse=True)
    return streaming[0]['path']
  elif images:
    images.sort(key=lambda x:x['resolution'][0], reverse=True)
    return images[0]['path']
  else:
    return None



def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def job_launcher(api, parallel_capture, capture_process, media_inputs):
  current_jobs=[]
  for media_input in media_inputs:

    # Only N jobs to run in parallel
    while len(current_jobs) >= parallel_capture:
      for job_idx, (job, media_id) in enumerate(current_jobs):
        if job.poll() is not None:
          del current_jobs[job_idx]
      time.sleep(0.050) # check all jobs every 50 ms

    # Compute real path based on media input
    if type(media_input) is str:
      ffmpeg_media_input = media_input
      api = None
      media_id = media_input
    else:
      if 'id' not in media_input:
        continue
      media_obj = api.get_media(media_input['id'], presigned=86400)
      ffmpeg_media_input = get_best_quality(media_obj)
      media_id = media_input['id']
      if ffmpeg_media_input is None:
        print(f"Can't process {media_obj.name} ({media_obj.id})")
        continue
      if media_obj.attributes.get('Object Detector Processed', 'No') != 'No':
        print(f"Skipping already processed file {media_obj.name} ({media_obj.id})")
        continue

    # construct the process invocation.
    this_media_capture = capture_process.copy()
    for idx,term in enumerate(this_media_capture):
        this_media_capture[idx] = term.replace('%{TIME}', datetime.datetime.utcnow().isoformat().replace(':','_'))
        this_media_capture[idx] = term.replace('%{INPUT}', ffmpeg_media_input)
    #print(' '.join(this_media_capture))
    capture = subprocess.Popen(this_media_capture, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    current_jobs.append((capture, media_id))

  # wait for last job to clean up
  while len(current_jobs) > 1:
    for job_idx, (job,media_id) in enumerate(current_jobs):
      if job.poll() is not None:
        del current_jobs[job_idx]
        print(f"Sent {media_id}")
    time.sleep(0.050) # check all jobs every 50 ms

def server_thread(buffer,free_queue, process_queue, dims, strategy):
  block_size=4*1024*1024
  expectedFrameSize = dims[0]*dims[1]*dims[2]
  serverSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
  # Sometimes on CTRL-C this doesn't get cleaned up.
  serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  serverSocket.bind(("localhost",strategy.get('port', 20000)))
  serverSocket.listen(100)

  api = get_tator_api(strategy)
  capture_process_str = strategy['capture'].get('capture_process', None)
  parallel_capture = strategy['capture'].get('parallel', 1)
  media_inputs = strategy['capture'].get('inputs', None)
  if os.getenv("TATOR_MEDIA_IDS"):
    id_list = os.getenv("TATOR_MEDIA_IDS").split(',')
    media_inputs = [{'id': int(x)} for x in id_list]

  # Blow out sections to individual media
  for media_input in media_inputs:
    if 'section_id' in media_input:
      section_obj = api.get_section(media_input['section_id'])
      media_objs = api.get_media_list(section_obj.project, section=section_obj.id)
      print(f"Section {section_obj.name} has {len(media_objs)}")
      new_media_objs = [{'id': x.id} for x in media_objs]
      media_inputs.extend(new_media_objs)

  job_launcher_t = threading.Thread(target=job_launcher, args=(api, parallel_capture, capture_process_str, media_inputs))
  job_launcher_t.start()
  print(f"About to process {len(media_inputs)}")
  media_begin = time.time()
  for media_count,media_input in enumerate(media_inputs):
    bytes_received = 0
    if type(media_input) is str:
      unique_media_id = media_input
    else:
      # This logic is required to keep the number of accept calls
      # the expected value
      if 'id' not in media_input:
        continue
      media_obj = api.get_media(media_input['id'])
      ffmpeg_media_input = get_best_quality(media_obj)
      unique_media_id = media_input['id']
      if ffmpeg_media_input is None:
        print(f"Can't process {media_obj.name} ({media_obj.id})")
        continue
      if media_obj.attributes.get('Object Detector Processed', 'No') != 'No':
        print(f"Skipping already processed file {media_obj.name} ({media_obj.id})")
        continue

    conn, _ =  serverSocket.accept() # Ctrl-C is blocked here. Kind of annoying.
    frame_count = 0
    current_buffer = free_queue.get()
    frame_buf = buffer[current_buffer]
    begin = time.time()
    this_image = 0
    # Receive up to the block size without going over an image boundary
    data_left = min(block_size,expectedFrameSize-this_image)

    # receive data right into a memory buffer only share idx over queues
    got_bytes = conn.recv_into(memoryview(frame_buf)[this_image:this_image+data_left], data_left)
    while got_bytes:
      bytes_received += got_bytes
      this_image += got_bytes
      if this_image == expectedFrameSize:
        this_image = 0
        process_queue.put((current_buffer, unique_media_id, frame_count))
        frame_count += 1
        current_buffer = free_queue.get() #get next available buffer
        frame_buf = buffer[current_buffer]
      data_left = min(block_size,expectedFrameSize-this_image)
      got_bytes = conn.recv_into(memoryview(frame_buf)[this_image:this_image+data_left], data_left)

    # If buffer isn't totally used return it, else we leak 1 buffer per media
    if this_image != expectedFrameSize:
      free_queue.put(current_buffer)
    end = time.time()
    duration = end - begin
    time_per_frame = duration / frame_count
    fps = 1.0 / time_per_frame
    print(f"media {media_count} ({unique_media_id}): bytes= {bytes_received}, duration = {duration}, frames={frame_count}, FPS = {fps}")
    if (media_count + 1) % 10 == 0:
      media_now = time.time()
      print(f"Summary: Total media={media_count+1}, total_time = {media_now-media_begin}, avg = {(media_now-media_begin)/(media_count+1)}")
    # Set sentinel flag to avoid duping boxes.
    if api:
      try:
        api.update_media(media_obj.id, {'attributes': {"Object Detector Processed": datetime.datetime.now().isoformat()}})
      except:
        pass

  # End the downchain processing send frame 0 to make printouts look good.
  job_launcher_t.join()
  process_queue.put((None,None,-1))

batch_results=[]
def save_thread(save_queue, strategy):
  results = save_queue.get()
  if strategy['save'].get('file', None):
    fp = open(strategy['save'].get('file', None), 'w')
    fp.write(f"Media, Frame,x1,y1,x2,y2,score,label\n")
  else:
    fp = None
  names = strategy['detector']['names']
  dims = strategy['detector']['size']


  tator_config = strategy['save'].get('tator', None)
  if tator_config:
    api = get_tator_api(strategy)
    localization_type_id = tator_config['localization_type_id']
    version_id = tator_config.get('version_id',None)
    project_id = api.get_localization_type(localization_type_id).project
    score_mapping = tator_config.get('mapping',{}).get('score','Confidence')
    label_mapping = tator_config.get('mapping',{}).get('label','Species')
    upload_batch_size = tator_config.get('upload_batch_size', 500)
  else:
    api = None

  # functor to periodically send batch results + clean out
  def add_to_batch(datum):
    global batch_results
    if api is None:
      return
    if datum:
      batch_results.append(datum)
    if len(batch_results) > upload_batch_size or datum is None:
      print(f"Uploading a batch of len(batch_results)")
      for response in tator.util.chunked_create(api.create_localization_list,
                                                project_id,
                                                localization_spec=batch_results):
            pass
      batch_results=batch_results[upload_batch_size:]
  while results is not None:
    for box,score,label_id in zip(results['boxes'], results['scores'], results['classes']):
      frame = results['frame']
      media = results['media']
      x0 = box[0]/dims[1]
      y0 = box[1]/dims[0]
      x1 = box[2]/dims[1]
      y1 = box[3]/dims[0]
      if api:
        new_object={'x': x0, 'y': y0, 'width': x1-x0, 'height': y1-y0, score_mapping: score, label_mapping: names[label_id], 'media_id': media, 'frame': frame, 'type': localization_type_id}
        if version_id:
          new_object.update({'version': version_id})
        add_to_batch(new_object)
      if fp:
        data=f"{media}, {frame},{x0},{y0},{x1},{y1},{score},{names[label_id]}\n"
        fp.write(data)
        # flush out to disk after each write
        if strategy['save'].get('flush', True):
          fp.flush()

    results = save_queue.get()
  # Flush the rest of the results
  add_to_batch(None)

def broadcast_thread(buffers, send_queue, free_queue, strategy):
  if strategy.get('broadcast_process', None):
    print("Lanching broadcast process")
    broadcast = subprocess.Popen(strategy['broadcast_process'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  else:
    broadcast = None

  def exit_handler(signum, _):
    broadcast.kill()
    exit(0)

  signal.signal(signal.SIGINT, exit_handler)
  signal.signal(signal.SIGTSTP, exit_handler)

  clientSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
  connected = False
  while connected == False:
    try:
      clientSocket.connect(("127.0.0.1", 20001))
      connected = True
    except:
      print("Could not connect to broadcast port")
      time.sleep(0.25)
  frames = 0
  total_frames = 0
  # processing allowance in nanoseconds
  allowance = 1.0/29.97 * 1000000000
  begin_batch = time.time()
  while True:
    current_buffer = send_queue.get()
    while send_queue.qsize() < 4:
      pass
    begin = time.time_ns()
    while current_buffer is not None:
      frame_buf = buffers[current_buffer]
      frames += 1
      total_frames += 1
      clientSocket.send(frame_buf)
      free_queue.put(current_buffer)
      current_buffer = None
      if send_queue.qsize() > 16:
        while (time.time_ns() - begin) < allowance*0.95:
          pass
      else:
        while (time.time_ns() - begin) < allowance:
          pass
      while current_buffer is None:
        try:
          current_buffer = send_queue.get_nowait()
        except:
          current_buffer = None
      if frames % 100 == 0:
        duration = time.time() - begin_batch
        time_per_frame = duration / 100
        fps = 1.0 / time_per_frame
        print(f"processed_frames={total_frames}, FPS = {fps}, Depth= {free_queue.qsize()}, {send_queue.qsize()}")
        #reset
        frames = 0
        begin_batch = time.time()
      begin = time.time_ns()

def drawBox(bgr, box, score, label, scale=0.67,margin=10):
  cv2.rectangle(bgr, box[0:2], box[2:4], color=(208,224,64), thickness=3)
  msg = f"{label}, {round(score,2)}"
  labelSize = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
  x1 = box[0]
  y1 = box[1] - labelSize[0][1]-margin
  x2 = x1 + labelSize[0][0]
  y2 = box[1] - margin
  #cv2.rectangle(bgr, (x1,y1), (x2,y2), color=(0,0,0,0.25), thickness=cv2.FILLED)
  # blend alpha instead of drawing solid black
  fill_rect = np.ones_like(bgr[y1:y2,x1:x2,:])
  fill_rect[:,:] = np.array((0,0,0))
  bgr[y1:y2,x1:x2,:] = cv2.addWeighted(bgr[y1:y2,x1:x2,:],0.75,fill_rect,0.25,0.0)
  cv2.putText(bgr, msg, (x1,y2), cv2.FONT_HERSHEY_DUPLEX, scale,color=(255,255,255),thickness=1,lineType=cv2.LINE_AA)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, default=20000)
  parser.add_argument('--cpu-only',
                      action='store_true')
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument("strategy_file")
  args = parser.parse_args()

  # Schema for strategy file
  # Each key has a value specfying comment and a boolean as to whether
  # it is required
  strategy_schema= {
    'capture_process': ('If present run this process after lauching capture thread',False),
    'broadcast_process': ('If present run this process before starting capture', False),
    'detector': {
      'size': ('Array representing size of network [H,W]', True),
      'threshold': ('Keep threshold', True),
      'names' : ('Array of class names', True)
    }
  }



  with open(args.strategy_file, 'r') as fp:
    strategy = yaml.safe_load(fp)

  dims=[*strategy['detector']['size'],3]
  frame_interval = strategy['detector'].get('interval',1)
  num_buffers = 64
  free_queue = multiprocessing.Queue(num_buffers)
  process_queue = multiprocessing.Queue(num_buffers)

  buffers=[]
  for x in range(num_buffers):
    buffers.append(RawArray(ctypes.c_uint8, dims[0]*dims[1]*dims[2]))
    free_queue.put(x)

  backbone = strategy['detector'].get('backbone', 'COCO-Detection/retinanet_R_50_FPN_3x.yaml')
  config = strategy['detector']['config']
  weights = strategy['detector']['weights']

  # Make a temporary work dir, or fetch from pipeline arguments
  temp_work_dir=os.getenv("TATOR_WORK_DIR")
  temp_gc = None
  if temp_work_dir is None:
    temp_gc = tempfile.TemporaryDirectory()
    temp_work_dir = temp_gc.name

  def handle_potential_fetch(path):
    # If config/weights are URLs, download them locally
    if path.startswith('http://') or path.startswith('https://'):
      local_name = path.split('/')[-1]
      local_path = os.path.join(temp_work_dir, local_name)
      print(f"Downloading '{path}' to '{local_path}'")
      download_file(path, local_path)
      return local_path
    else:
      return path

  config = handle_potential_fetch(config)
  weights = handle_potential_fetch(weights)

  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file(backbone))
  cfg.merge_from_file(config)
  cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.3
  cfg.MODEL.WEIGHTS = weights # path to the model file
  model = build_model(cfg)
  checkpointer = DetectionCheckpointer(model)
  checkpointer.load(cfg.MODEL.WEIGHTS)
  model = model.eval().cuda()

  # Seperate NMS
  model_nms = torchvision.ops.nms
  nms_threshold = strategy['detector'].get('nms_threshold', 0.55)

  if strategy.get('broadcast_process',None):
    print("Broadcast mode enabled")
    publish_queue = multiprocessing.Queue(num_buffers)
    broadcast = multiprocessing.Process(None, broadcast_thread, args=(buffers, publish_queue,free_queue, strategy))
    broadcast.start()
  else:
    publish_queue = None

  if strategy.get('save',None):
    print("Save mode enabled")
    save_queue = multiprocessing.Queue(num_buffers)
    save = multiprocessing.Process(None, save_thread, args=(save_queue, strategy))
    save.start()
  else:
    save_queue = None

  server = multiprocessing.Process(None, server_thread, args=(buffers, free_queue, process_queue, dims, strategy))
  server.start()

  print("Loaded model")

  buffer_idx, media_id, frame_count  = process_queue.get()

  names = strategy['detector']['names']

  current={"boxes": [], "scores": [], "classes":[]}
  begin = time.time()
  while buffer_idx is not None:
    if frame_count % frame_interval == 0:
      bgr = np.frombuffer(buffers[buffer_idx],
                          dtype=np.uint8).reshape(dims)
      blob = torch.as_tensor(
                bgr.transpose(2,0,1)).cuda()
      results = model([{"image":blob}])

      # Process results on CPU
      cpu_results = results[0]
      cpu_results["instances"] = cpu_results["instances"][
        model_nms(
                cpu_results["instances"].pred_boxes.tensor,
                cpu_results["instances"].scores,
                nms_threshold,
            )
            .to("cpu")
            .tolist()
      ]
      instance_dict = cpu_results["instances"].get_fields()
      pred_boxes = instance_dict["pred_boxes"]
      scores = instance_dict["scores"]
      pred_classes = instance_dict["pred_classes"]
      current={"boxes": [], "scores": [], "classes":[]}
      for box, score, cls in zip(pred_boxes, scores, pred_classes):
          if score > strategy['detector']['threshold']:
            current['boxes'].append(np.array(box.tolist(),dtype=np.uint32))
            current['scores'].append(score.tolist())
            current['classes'].append(cls.tolist())
      if current['boxes']:
        if save_queue:
          current['frame'] = frame_count
          current['media'] = media_id
          save_queue.put(current)
        if publish_queue:
          for box,score,label_id in zip(current['boxes'], current['scores'], current['classes']):
            drawBox(bgr, box, score, names[label_id])
            frame_data = bgr.tobytes()
      if publish_queue:
        publish_queue.put(buffer_idx)
      else:
        free_queue.put(buffer_idx)
    else:
      if current['boxes'] and publish_queue:
        bgr = np.frombuffer(buffers[buffer_idx],
                            dtype=np.uint8).reshape(dims)
        for box,score,label_id in zip(current['boxes'], current['scores'], current['classes']):
          drawBox(bgr, box, score, names[label_id])
          frame_data = bgr.tobytes()
      if publish_queue:
        publish_queue.put(buffer_idx)
      else:
        free_queue.put(buffer_idx)
    buffer_idx, media_id, frame_count = process_queue.get()

    if args.verbose:
      if (frame_count+1) % 100 == 0:
        duration = time.time() - begin
        time_per_frame = duration / 100
        fps = 1.0 / time_per_frame
        print(f"total_frames={frame_count}, graph time={time_per_frame},graph fps = {fps}, Depth= {free_queue.qsize()}")
        begin = time.time()

  server.join()
  if save_queue:
    save_queue.put(None)
    save.join()
  if publish_queue:
    save_queue.put(None)
    broadcast.join()



if __name__=="__main__":
  main()
