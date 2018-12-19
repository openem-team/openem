"""Generic utility functions."""

import numpy as np
import math
import skimage
from skimage.transform import AffineTransform

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
    return skimage.transform.warp(full_rgb, tform, mode='edge', order=order, output_shape=(out_size, out_size))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l) // n * n + n - 1, n):
        if len(l[i:i + n]):
            yield l[i:i + n]

