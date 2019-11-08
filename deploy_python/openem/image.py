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
