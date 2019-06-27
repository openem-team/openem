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

"""Utilities for image augmentation."""

import numpy as np
from scipy.misc import imresize

from skimage import io,transform
import matplotlib.pyplot as plt
import numpy as np

def grayscale(rgb):
    """Converts image to grayscale.
    """
    return rgb.dot([0.299, 0.587, 0.114])


def saturation(rgb, variance=0.5, mean=None):
    """Modifies saturation by a random amount.
    """
    if mean is None:
        mean = np.random.random()
    gray = grayscale(rgb)
    alpha = 2 * mean * variance
    alpha += 1 - variance
    rgb = rgb * alpha + (1 - alpha) * gray[:, :, None]
    return np.clip(rgb, 0, 1.0)


def brightness(rgb, variance=0.5, mean=None):
    """Modifies brightness by a random amount.

    # Arguments
        img: Image to modify.
        variance: Variance in brightness.
        mean: Mean in brightness, randomly generated if None.

    # Returns
        Modified image.
    """
    if mean is None:
        mean = np.random.random()

    alpha = 2 * mean * variance
    alpha += 1 - variance
    rgb = rgb * alpha
    return np.clip(rgb, 0, 1.0)


def contrast(rgb, variance=0.5, mean=None):
    """Modifies contrast by a random amount.

    # Arguments
        img: Image to modify.
        variance: Variance in contrast.
        mean: Mean in contrast, randomly generated if None.

    # Returns
        Modified image.
    """
    if mean is None:
        mean = np.random.random()

    gray = grayscale(rgb).mean() * np.ones_like(rgb)
    alpha = 2 * mean * variance
    alpha += 1 - variance
    rgb = rgb * alpha + (1 - alpha) * gray
    return np.clip(rgb, 0, 1.0)


def lighting(img, variance=0.5, mean=None):
    """Modifies lighting by a random amount.

    # Arguments
        img: Image to modify.
        variance: Variance in lighting.
        mean: Mean in lighting, randomly generated if None.

    # Returns
        Modified image.
    """
    if mean is None:
        mean = np.random.randn(3)

    orig = img.copy
    cov = np.cov(img.reshape(-1, 3) / 1.0, rowvar=False)
    eigval, eigvec = np.linalg.eigh(cov)
    noise = mean * variance
    noise = eigvec.dot(eigval * noise) * 1.0
    img += noise
    try:
        img += noise
        return np.clip(img, 0, 1.0)
    except TypeError:
        return orig


def horizontal_flip(img):
    """Flips an image horizontally.
    """
    return img[:, ::-1]


def vertical_flip(img):
    """Flips an image vertically.
    """
    return img[::-1]


def blurred_by_downscaling(img, ratio):
    """Applies blur to an image by downscaling it.

    # Arguments
        img: Image to modify.
        ratio: Ratio to downscale by.

    # Returns
        Blurred image.
    """
    resampling = np.random.choice(['nearest', 'lanczos', 'bilinear', 'bicubic'])

    if ratio == 1:
        return img

    width = img.shape[1]
    height = img.shape[0]
    small = imresize(img, ratio, interp=resampling)
    large = imresize(small, size=(height, width), interp='bilinear').astype(np.float32)/255.0
    return large

def resizeAndFill(image, desired_shape):
    """
    Resize an image to a desired shape (height,width) and maintaining
    the aspect ratio. Fill any extra areas with black.
    :param image: ndarray of the image
    :param desired_shape: tuple describing the output shape

    :returns image_resized,scaleFactor:
    image_resized is ndarray represented the scaled + padded image
    scaleFactor Given a pixel coordinate in the original this factor should
    be applied to land on the new image.
    """
    image_height=image.shape[0]
    image_width=image.shape[1]
    image_channels=image.shape[2]

    desired_width=desired_shape[1]
    desired_height=desired_shape[0]
    growth_factor=min(desired_height/image_height,
                           desired_width/image_width)
    new_height=int(image_height*growth_factor)
    new_width=int(image_width*growth_factor)
    image_resized=transform.resize(image,(new_height,new_width),anti_aliasing=True)
    added_rows=desired_height-new_height
    added_cols=desired_width-new_width

    if added_rows:
        black_bar=np.zeros((added_rows, new_width, image_channels))
        image_resized = np.append(image_resized,black_bar, axis=0)

    if added_cols:
        black_bar=np.zeros((new_height, added_cols, image_channels))
        image_resized = np.append(image_resized,black_bar, axis=1)

    scaleFactor=(float(new_height)/image_height,float(new_width)/image_width)
    return image_resized, scaleFactor
