""" Wrapper to invoke retinanet training scripts from openem """

import subprocess
import os
import csv
import pandas as pd
import cv2
import numpy as np

from collections import namedtuple
from pprint import pprint

from openem_train.util import utils
from openem_train.util.roi_transform import RoiTransform
from openem_train.util.img_augmentation import resizeAndFill

import progressbar

FishBoxDetection = namedtuple(
    'FishBoxDetection',
    ['video_id', 'frame', 'x', 'y', 'width', 'height', 'theta', 'class_id'])

def prep(config):
    """ Generates a csv file compatible with retinanet training script
        outputs it in the OPENEM_WORK area for subsequent commands to use
    """

    work_dir = config.work_dir()
    retinanet_dir = os.path.join(work_dir, "retinanet")
    species_csv = os.path.join(retinanet_dir, "species.csv")
    retinanet_csv = os.path.join(retinanet_dir, "totalPopulation.csv")

    os.makedirs(retinanet_dir, exist_ok=True)

    # Generate the species csv file first
    # This is a csv file with each species on a new line, with no
    # header
    species=[]
    for idx,name in enumerate(config.species()):
        species.append({'species': name,
                        'id': idx})
    species_df = pd.DataFrame(columns=['species', 'id'], data=species)
    species_df.to_csv(species_csv, header=False, index=False)

    # Generate the annotations csv for retinanet; this is in the format
    # from the keras_retinanet.preprocessing.csv_generator module
    # img_file, x1, y1, x2, y2, class_name = row
    # Where x1,y1 and x2,y2 represent the diagonal across a box annotation
    # Also valid row is
    # image_file,,,,,
    # This represents an image file with no annotations; it is ignored by
    # the preprocessor in keras_retinanet

    # Before we start converting/transforming we setup the roi transform
    # object and figure out if we are in line or box mode
    roi_transform = RoiTransform(config)
    length = pd.read_csv(config.length_path())

    keys = length.keys()
    linekeys=['x1','x2','y1','y2']
    boxkeys=['x','y','width','height','theta']
    if all(x in keys for x in linekeys):
        lineMode = True
    elif all(x in keys for x in boxkeys):
        lineMode = False

    retinanet_cols=['img_file', 'x1', 'y1', 'x2', 'y2', 'class_name']
    retinanet_df = pd.DataFrame(columns=retinanet_cols)

    bar = progressbar.ProgressBar(max_value=len(length), redirect_stdout=True)
    # Iterate over each row in the length.csv and make a retinanet.csv
    for sample, row in bar(length.iterrows()):
        # Ignore no detections for retinanet csv
        if row.species_id == 0:
            continue


        # Construct image path (either png or jpg)
        jpg_image_file = os.path.join(config.train_rois_dir(),
                                      row.video_id,
                                      f"{row.frame:04d}.jpg")
        png_image_file = os.path.join(config.train_rois_dir(),
                                      row.video_id,
                                      f"{row.frame:04d}.png")

        if os.path.exists(jpg_image_file):
            image_file = jpg_image_file
        elif os.path.exists(png_image_file):
            image_file = png_image_file

        img_data = cv2.imread(image_file)

        # TODO: Don't use detection width/height here, needs to be in
        # image coordinates...
        tform = roi_transform.transform_for_clip(
            row.video_id,
            dst_w=img_data.shape[1],
            dst_h=img_data.shape[0])

        # Species id in openem is 1-based index
        species_id_0 = row.species_id - 1
        species_name = config.species()[species_id_0]

        # OpenEM detection csv is in image coordinates, need to convert
        # that to roi coordinates because that is what we train on.
        # Logic is pretty similar for line+aspect ratio and box style
        # annotations
        if lineMode:
            aspect_ratio = config.aspect_ratios()[species_id_0]
            coords_image = np.array([[row.x1,
                                      row.y1],
                                     [row.x2,
                                      row.y2]])
            if tform:
                coords_roi = tform.inverse(coords_image)
            else:
                # There is no transform
                coords_roi = coords_image
            coords_box0, coords_box1 = utils.bbox_for_line(coords_roi[0,:],
                                                           coords_roi[1,:],
                                                           aspect_ratio)
        else:
            # Make the row a detection object (in image coords)
            detection_image = FishBoxDetection(
                video_id=row.video_id,
                frame=row.frame,
                x=row.x, y=row.y,
                width=row.width,
                height=row.height,
                theta=row.theta,
                class_id=row.species_id
            )
            rotated_detection_image = utils.rotate_detection(detection_image)
            # Box is now converted from x,y,w,h to 4 points representing each
            # corner
            # We translate all 4 points
            if tform:
                coords_roi = tform.inverse(rotated_detecion_image)
            else:
                # There is no transform
                coords_roi = rotated_detection_image

            topLeftIdx,bottomRightIdx=utils.find_corners(coords_roi)
            # These are now the diagnol representing the bounding box.
            coords_box0=coords_roi[topLeftIdx]
            coords_box1=coords_roi[bottomRightIdx]


        # Coords are ints for retinanet
        coords_box0=np.round(coords_box0).astype(np.int)
        coords_box1=np.round(coords_box1).astype(np.int)

        zero=np.array([0,0])
        negative=np.max([coords_box0 < zero, coords_box1 < zero])

        if negative == True:
            print(f"WARNING:\tLocalization went off ROI, line {sample}")
            print(f"\t\t {row.video_id}, frame={row.frame}")
            continue

        # Make the datum and append it to the big dataframe
        datum={'img_file' : image_file,
               'class_name' : species_name,
               'x1': coords_box0[0],
               'y1': coords_box0[1],
               'x2': coords_box1[0],
               'y2': coords_box1[1]}

        retinanet_df = retinanet_df.append(pd.DataFrame(columns=retinanet_cols,
                                                        data=[datum]))

    retinanet_df.to_csv(retinanet_csv, index=False, header=False)

def getPopulationStats(config, df):
    stats={}
    total=len(df)
    for name in config.species():
        count = len(df[df.class_name==name])
        stats[name] = (count, count/total)
    return stats

def split(config):
    work_dir = config.work_dir()
    retinanet_dir = os.path.join(work_dir, "retinanet")
    totalPopulation_csv = os.path.join(retinanet_dir, "totalPopulation.csv")
    annotations_csv = os.path.join(retinanet_dir, "annotations.csv")
    validation_csv = os.path.join(retinanet_dir, "validation.csv")
    retinanet_cols=['img_file', 'x1', 'y1', 'x2', 'y2', 'class_name']
    total_df = pd.read_csv(totalPopulation_csv,
                           header=None,
                           names=retinanet_cols)
    if config.detect_do_validation():
        random_seed = config.detect_val_random_seed()
        val_population = config.detect_val_population()
        print(f"Generating validation data set {val_population*100}% @ RS:{random_seed}")
        train_pop = 1.0 - val_population

        videos_list=total_df['img_file'].unique()
        videos_df=pd.DataFrame(columns=['img_file'],
                               data=videos_list)
        train_vids=videos_df.sample(frac=train_pop,
                                      random_state=random_seed)
        train_df = total_df.loc[total_df['img_file'].isin(train_vids["img_file"].tolist())]
        validation_df=total_df.drop(train_df.index)

        print("Total Population:")
        pprint(getPopulationStats(config, total_df))
        print("Train Population:")
        pprint(getPopulationStats(config, train_df))
        print("Validation Population:")
        pprint(getPopulationStats(config, validation_df))
        train_df.to_csv(annotations_csv, index=False, header=False)
        validation_df.to_csv(validation_csv, index=False, header=False)
    else:
        print("NOITICE: Not configured to do validation")
        print("Total Population:")
        pprint(getPopulationStats(config, total_df))
        total_df.to_csv(annotations_csv, index=False, header=False)

def train(config):
    work_dir = config.work_dir()
    retinanet_dir = os.path.join(work_dir, "retinanet")
    annotations_csv = os.path.join(retinanet_dir, "annotations.csv")
    validation_csv = os.path.join(retinanet_dir, "validation.csv")
    species_csv = os.path.join(retinanet_dir, "species.csv")
    snapshot_dir = os.path.join(retinanet_dir, "train_snapshots")
    log_dir = os.path.join(retinanet_dir, "train_log")
    if not os.path.exists(species_csv):
        print(f"Need to make species.csv in {work_dir}")
        print("Attempting to generate it for you from config.ini")
        with open(species_csv,'w') as csv_file:
            writer = csv.writer(csv_file)
            for species in config.species():
                print(f"\t+Adding {species}")
                writer.writerow([species])
            print("Done!")
    else:
        print("Detected Species.csv in training dir")

    train_annotations=pd.read_csv(annotations_csv, header=None, names=['vid_id', 'x1','y1','x2','y2', 'species'])
    unique_videos = train_annotations['vid_id'].unique()
    steps_per_epoch = len(unique_videos) / config.detect_batch_size()
    steps_per_epoch = int(np.floor(steps_per_epoch))
    print("Calculated steps per epoch = {steps_per_epoch}")
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    args = ['python',
            '/keras_retinanet/scripts/train.py',
            '--train-img-dir',
            config.train_rois_dir(),
            '--batch-size',
            str(config.detect_batch_size()),
            '--snapshot-path',
            snapshot_dir,
            '--log-dir',
            log_dir,
            '--epochs',
            str(config.detect_num_epochs()),
            '--steps-per-epoch',
            str(steps_per_epoch)
    ]

    backbone = config.detect_backbone()
    if backbone:
        args.extend(['--backbone',
                     backbone])

    args.extend(['csv',
                 annotations_csv,
                 species_csv])
    if config.detect_do_validation():
        args.extend(['--val-annotations',
                     validation_csv])
    args.extend(['--image_min_side',
                 str(config.detect_height()),
                 '--image_max_side',
                 str(config.detect_width())])
    force_aspect = config.detect_force_aspect()
    if force_aspect:
        args.extend(['--force-aspect-ratio',
                     str(force_aspect)])

    cmd = " ".join(args)
    print(f"Command = {cmd}")
    p=subprocess.Popen(args)
    p.wait()
    return p.returncode

def tensorboard(config):
    work_dir = config.work_dir()
    retinanet_dir = os.path.join(work_dir, "retinanet")
    log_dir = os.path.join(retinanet_dir, "train_log")
    port = config.tensorboard_port()

    args=['tensorboard',
          '--logdir',
          log_dir,
          '--port',
          str(port)]
    p=subprocess.Popen(args)
    p.wait()
    return p.returncode

def predict(config):
    import openem
    from openem.Detect import Detection,RetinaNet
    import pandas as pd
    import cv2

    image_dims = (config.detect_height(), config.detect_width())
    retinanet = RetinaNet.RetinaNetDetector(config.detect_retinanet_path(), imageShape=image_dims)
    limit = None
    count = 0
    threshold=0
    if config.config.has_option('Detect', 'Limit'):
        limit = config.config.getint('Detect','Limit')
    if config.config.has_option('Detect', 'Threshold'):
        threshold= config.config.getfloat('Detect', 'Threshold')

    print(f"Using threshold {threshold} for up to {limit} files")

    result_csv = config.detect_inference_path()
    result_cols=['video_id',
                  'frame',
                  'x','y','w','h',
                  'det_conf','det_species']
    result_df = pd.DataFrame(columns=result_cols)
    result_df.to_csv(result_csv, header=True, index=False)
    bar = progressbar.ProgressBar(redirect_stdout=True)
    # TODO: Use test images here?
    for img_path in bar(config.train_rois()):
        path, f = os.path.split(img_path)
        frame, _ = os.path.splitext(f)
        frame=int(frame)
        video_id = os.path.basename(os.path.normpath(path))
        img = cv2.imread(img_path)
        retinanet.addImage(img)
        results = retinanet.process(threshold, frame=frame, video_id=video_id)
        image_results=results[0]
        for result in image_results:
            datum = {'video_id': result.video_id,
                     'frame': result.frame,
                     'x': result.location[0],
                     'y': result.location[1],
                     'w': result.location[2],
                     'h': result.location[3],
                     'det_species': result.species,
                     'det_conf': result.confidence}
            record = pd.DataFrame(columns=result_cols,
                                  data=[datum])
            record.to_csv(result_csv, header=False, index=False, mode='a')
