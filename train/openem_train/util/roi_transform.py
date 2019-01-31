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

"""Defines RoiTransform class."""

from collections import namedtuple
import pandas as pd
import numpy as np
from skimage.transform import SimilarityTransform

RulerPoints = namedtuple('RulerPoints', ['x1', 'y1', 'x2', 'y2'])

class RoiTransform:
    """Loads ruler points and computes transforms from it.
    """
    def __init__(self, config):
        """Constructor.

        # Arguments
            config: ConfigInterface object.
        """
        self.ruler_points = {}
        ruler_points = pd.read_csv(config.find_ruler_inference_path())
        for _, row in ruler_points.iterrows():
            self.ruler_points[row.video_id] = RulerPoints(
                x1=row.x1,
                y1=row.y1,
                x2=row.x2,
                y2=row.y2)

    def transform_for_clip(self, video_id, dst_w=720, dst_h=360, points_random_shift=0):
        """Finds transform to crop around ruler.

        # Arguments
            video_id: Video ID.
            dst_w: Width of cropped image.
            dst_h: Height of cropped image.
            points_random_shift: How many points to randomly shift image.
        """
        img_points = np.array([[dst_w * 0.1, dst_h / 2], [dst_w * 0.9, dst_h / 2]])
        points = self.ruler_points[video_id]

        ruler_points = np.array([[points.x1, points.y1], [points.x2, points.y2]])

        if points_random_shift > 0:
            img_points += np.random.uniform(-points_random_shift, points_random_shift, (2, 2))

        tform = SimilarityTransform()
        tform.estimate(dst=ruler_points, src=img_points)

        return tform

