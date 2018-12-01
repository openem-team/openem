"""Generic utility functions."""

import numpy as np

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
