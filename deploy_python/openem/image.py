""" Utility module for dealing with image operations """

import numpy as np
import math
import cv2

def crop(image, roi):
    """ Returns a *copy* of the region of interest from the image
    image: ndarray
           Represents image data
    roi: tuple
         (x,y,w,h) tuple -- presumably from openem.FindRuler.findRoi
    """
    x0=int(roi[0])
    y0=int(roi[1])
    x1=int(roi[0]+roi[2])
    y1=int(roi[1]+roi[3])
    cropped=np.copy(image[y0:y1,x0:x1])
    return cropped

def resize_and_fill(image, desired_shape):
    """
    Resize an image to a desired shape (height,width) and maintaining
    the aspect ratio. Fill any extra areas with black.

    :param image: ndarray of the image
    :param desired_shape: tuple describing the output shape

    :returns image_resized,scaleFactor:

    image_resized is ndarray represented the scaled + padded image

    scaleFactor Given a pixel coordinate in the original this factor
                should be applied to land on the new image.

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
    # cv2 arguments are backwards from other libraries here, width is first (col,rows)
    image_resized=cv2.resize(image,(new_width, new_height))
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

def force_aspect(image, required_aspect_ratio):
    """ Given an image; force it to be given aspect ratio prior to resizing """
    img_height = image.shape[0]
    img_width = image.shape[1]
    img_aspect = img_width / img_height
    if math.isclose(required_aspect_ratio, img_aspect):
        return image

    if img_aspect < required_aspect_ratio:
        # This is when the image is boxier than the aspect ratio
        # so we add a black bar at the right side to compensate
        # this added bar does not effect annotation coordinates
        new_img_width = round(img_height * required_aspect_ratio)
        image,sf = resize_and_fill(image, (img_height, new_img_width))
    else:
        # This is when the image is narrower than the aspect ratio
        # so we add a black bar at the bottom to compensate
        # this added bar does not effect annotation coordinates
        new_img_height = round(img_width / required_aspect_ratio)
        image,sf = resize_and_fill(image, (new_img_height, img_width))
    assert math.isclose(sf[0],1.0) and math.isclose(sf[1],1.0)
    return image
