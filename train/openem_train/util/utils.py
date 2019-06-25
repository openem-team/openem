__copyright__ = "Copyright (C) 2018 CVision AI."
__license__ = "GPLv3"
# This file is part of OpenEM, released under GPLv3.
# OpenEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenEM.  If not, see <http://www.gnu.org/licenses/>.

"""Generic utility functions."""

import os
import glob
import math
import numpy as np
import skimage
from skimage.transform import AffineTransform
import math

def find_corners(coords):
    """ Find the left and right corner of the given box
    :param coords: 4-element array-type
    :returns idx of Left Corner
    """
    minimum=np.min(coords, axis=0)
    maximum=np.max(coords, axis=0)
    minDistances=np.zeros(4)
    maxDistances=np.zeros(4)
    for idx,coord in enumerate(coords):
        minDistances[idx] = math.hypot(coord[1]-minimum[1],coord[0]-minimum[0])
        maxDistances[idx] = math.hypot(coord[1]-maximum[1],coord[0]-maximum[0])
    minIdx=np.argmin(minDistances)
    maxIdx=np.argmin(maxDistances)
    return (minIdx,maxIdx)

def rotate_detection(detection):
    """ Given a box detection, rotate it around point 0 by theta
        Points are in NE,SE,SW,NW ordering
    """
    Rot=np.array([[math.cos(detection.theta), -1.0*math.sin(detection.theta), 0],
                  [math.sin(detection.theta), math.cos(detection.theta), 0],
                  [0,0,1]])
    Translation=np.array([[1,0,detection.x],
                          [0,1,detection.y],
                          [0,0,1]])
    InverseTranslation=np.array([[1,0,-detection.x],
                                 [0,1,-detection.y],
                                 [0,0,1]])
    #Shift it, rotate it, shift it back
    RotTranslation=Translation.dot(Rot.dot(InverseTranslation));

    NorthEast=np.array([detection.x,detection.y])
    SouthEast=RotTranslation.dot(np.array([detection.x,detection.y+detection.height,1]))
    NorthWest=RotTranslation.dot(np.array([detection.x+detection.width,detection.y,1]))
    SouthWest=RotTranslation.dot(np.array([detection.x+detection.width,detection.y+detection.height,1]))

    SouthEast=SouthEast[:2]
    NorthWest=NorthWest[:2]
    SouthWest=SouthWest[:2]
    box=np.array([NorthEast,SouthEast,SouthWest,NorthWest])
    return box

def bbox_for_line(pt0, pt1, aspect_ratio=0.5):
    """Calculate bounding rect around box with line in the center

    # Arguments
        pt0: First point.
        pt1: Second point.
        aspect_ratio: rect aspect ratio, 0.5 - width == 0.5 line norm

    # Returns
        Bounding box.
    """
    diff = pt1 - pt0
    # vector perpendicular to p0-p1 with 0.5 aspect ratio norm
    perp = np.array([diff[1], -diff[0]]) * aspect_ratio * 0.5

    points = np.row_stack([pt0 + perp, pt0 - perp, pt1 + perp, pt1 - perp])
    return np.min(points, axis=0), np.max(points, axis=0)

def lock_layers_until(model, first_trainable_layer, verbose=False):
    """Locks layers until a given layer name.

    # Arguments
        model: Model to set trainable layers on.
        first_trainable_layer: Name of first trainable layer.
        verbose: True to print trainable status of each layer.
    """
    found_first_layer = False
    for layer in model.layers:
        if layer.name == first_trainable_layer:
            found_first_layer = True

        if verbose and found_first_layer and not layer.trainable:
            print('Make layer trainable:', layer.name)
            layer.trainable = True

        layer.trainable = found_first_layer

def get_image_crop(full_rgb, rect, scale_rect_x=1.0, scale_rect_y=1.0,
                   shift_x_ratio=0.0, shift_y_ratio=0.0,
                   angle=0.0, out_size=299, order=3):
    """Retrieves image crop.

    # Arguments
        full_rgb: Full image to crop.
        rect: Nominal rectangle to crop.
        scale_rect_x: Amount to scale the rect horizontally.
        scale_rect_y: Amount to scale the rect vertically.
        shift_x_ratio: Amount to shift rect horizontally.
        shift_y_ratio: Amount to shift rect vertically.
        angle: Rotation angle in degrees.
        out_size: Size of one side of output square.
        order: Order to use for transform.

    # Returns
        Cropped image.
    """
    center_x = rect.x + rect.w / 2
    center_y = rect.y + rect.h / 2
    size = int(max(rect.w, rect.h))
    size_x = size * scale_rect_x
    size_y = size * scale_rect_y

    center_x += size * shift_x_ratio
    center_y += size * shift_y_ratio

    scale_x = out_size / size_x
    scale_y = out_size / size_y

    out_center = out_size / 2

    tform = AffineTransform(translation=(center_x, center_y))
    tform = AffineTransform(rotation=angle * math.pi / 180) + tform
    tform = AffineTransform(scale=(1 / scale_x, 1 / scale_y)) + tform
    tform = AffineTransform(translation=(-out_center, -out_center)) + tform
    return skimage.transform.warp(
        full_rgb,
        tform,
        mode='edge',
        order=order,
        output_shape=(out_size, out_size)
    )

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l) // n * n + n - 1, n):
        if l[i:i + n]:
            yield l[i:i + n]

def get_best_detection(video_id, frame, dets):
    """Gets the best detection for a given video ID and frame.

    # Arguments:
        video_id: Video ID.
        frame: Frame number.
        dets: DataFrame containing detection data.

    # Returns:
        None if no detection found, best row otherwise.
    """
    same_vid = dets['video_id'] == video_id
    same_frame = dets['frame'] == frame
    matches = dets[same_vid & same_frame]

    if matches.shape[0] == 0:
        return None

    best = matches.sort_values('det_conf', ascending=False).iloc[0]
    if best.det_conf < 0.075:
        return None

    return best

def find_epoch(checkpoints_dir, epoch):
    """Finds checkpoint associated with the given epoch.

    # Arguments
        checkpoints_dir: Directory containing checkpoints.
        epoch: Epoch to find.

    # Returns
        Path to most recent checkpoint file for this epoch.
    """
    patt = os.path.join(
        checkpoints_dir,
        "checkpoint-{:03d}*".format(epoch))
    files = glob.glob(patt)
    if not files:
        msg = "Could not find checkpoint for epoch {}!"
        raise ValueError(msg.format(epoch))
    latest = max(files, key=os.path.getctime)
    return latest
