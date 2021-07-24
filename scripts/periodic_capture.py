import tator
# For custom upload routine
from tator.util._upload_file import _upload_file
from tator.transcode.make_thumbnails import make_thumbnails
from tator.transcode.transcode import make_video_definition
from tator.transcode.make_fragment_info import make_fragment_info
from tator.openapi.tator_openapi.models import CreateResponse

import argparse
import os
import subprocess
import json
import time
import datetime
import math
import pandas as pd
import uuid
import dateutil
import tempfile


def get_file_info(path):
  cmd = [
        "ffprobe",
        "-v","error",
        "-show_entries", "stream",
        "-print_format", "json",
        "-select_streams", "v:0",
        path,
    ]
  output = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
  video_info = json.loads(output)
  duration = video_info["streams"][0]['duration']
  r_framerate = video_info["streams"][0]['r_frame_rate'].split('/')
  fps = float(r_framerate[0])/float(r_framerate[1])

  start = os.path.splitext(os.path.basename(path))[0]
  start = dateutil.parser.parse(start.replace('_',':'))
  return start, float(duration), fps
  
def get_existing_chunks(api, args, project):
  section_list = api.get_section_list(project, name=args.section)
  if not section_list:
    print("Section doesn't exist")
    return []
  section_id = section_list[0].id
  media_list = api.get_media_list(project, section=section_id)
  return [x.name for x in media_list]

def upload_new_chunk(args, api, project, upload_gid, frameRange, chunk_name):
  media_response = api.create_media(project, {'gid': upload_gid,
                                              'name':chunk_name,
                                              'section': args.section,
                                              'type': args.media_type_id,
                                              'md5': tator.util.md5sum(args.video)})
  with tempfile.TemporaryDirectory() as td:
    temp_name = os.path.join(td, chunk_name)
    ffmpeg_args = ["ffmpeg",
                   "-i", args.video,
                   "-frames:v", str(frameRange[1]+1),
                   "-vf", f"select=between(n\\,{frameRange[0]}\\,{frameRange[1]})",
                   "-af", f"aselect=between(n\\,{frameRange[0]}\\,{frameRange[1]})",
                   "-c:v", "hevc_nvenc",
                   "-c:a", "aac",
                   "-preset", "hq",
                   "-2pass", "0",
                   "-rc-lookahead", "8",
                   "-rc", "vbr_hq",
                   "-cq", "30",
                   "-r", "29.97",
                   "-movflags", "faststart+frag_keyframe+empty_moov+default_base_moof",
                   "-y",
                   "-f", "mp4",
                   temp_name]
    print(" ".join(ffmpeg_args))
    p = subprocess.run(ffmpeg_args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for p,_ in tator.util.upload_media(api,
                                       args.media_type_id,
                                       temp_name,
                                       upload_gid=upload_gid,
                                       media_id=media_response.id):
      print(f"\r{p}%",end='', flush=True)
    data = pd.read_csv(args.metadata)
    this_chunk_data = data.loc[data['Frame'] >= frameRange[0]].loc[data['Frame'] <= frameRange[1]]
    print(f"\tThis chunk has {len(this_chunk_data)}")
    objs=[]
    for idx,row in this_chunk_data.iterrows():
      objs.append({'type': args.box_type_id,
                   'media_id': media_response.id,
                   'frame': row['Frame']-frameRange[0],
                   'x': row['x1'],
                   'y': row['y1'],
                   'height': row['y2']-row['y1'],
                   'width': row['x2']-row['x1'],
                   args.score_name: row['score'],
                   args.label_name: row['label']
                  })
    for response in tator.util.chunked_create(api.create_localization_list, project,
                                              localization_spec=objs):
      pass



def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--token", type=str, required=True)
  parser.add_argument("--media-type-id", type=int, required=True)
  parser.add_argument("--box-type-id", type=int, required=True)
  parser.add_argument("--section", type=str,default="broadcast")
  parser.add_argument("--label-name", type=str, default="Species")
  parser.add_argument("--score-name", type=str, default="Confidence")
  parser.add_argument("--frame-interval", type=int, default=30000)
  parser.add_argument("metadata")
  parser.add_argument("video")
  args = parser.parse_args()

  api = tator.get_api(token=args.token)
  media_type = api.get_media_type(args.media_type_id)
  project = media_type.project

  upload_gid = str(uuid.uuid1())
  
  while True:
    existing_chunks = get_existing_chunks(api, args, project)
    start,duration,fps = get_file_info(args.video)
    chunk_seconds = args.frame_interval/fps
    print(f"Uploading file in {args.frame_interval} frame chunks or {chunk_seconds} seconds.")
    print(f"File is {duration} @ FPS={fps}, start={start}")

    available_chunks = math.floor(duration/chunk_seconds)
    print(f"Existing={len(existing_chunks)}, Available={available_chunks}")
    for chunk_id in range(available_chunks):
      startFrame = chunk_id * args.frame_interval
      endFrame = ((chunk_id + 1) * args.frame_interval)-1
      offset = chunk_id * chunk_seconds
      chunk_start = start + datetime.timedelta(seconds=offset)
      print(f"{chunk_id}: {startFrame} to {endFrame} - {chunk_start}")
      chunk_name = f"{chunk_start}.mp4"
      if chunk_name in existing_chunks:
        print("\t Found existing chunk.")
      else:
        print("\t Uploading chunk")
        upload_new_chunk(args, api, project, upload_gid, (startFrame,endFrame), chunk_name)

    time.sleep(60)


if __name__=="__main__":
  main()