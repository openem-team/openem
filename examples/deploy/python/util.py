import numpy as np

def image_to_numpy(img):
    """Converts openem image to numpy array with appropriate shape.

    # Arguments
        img: openem Image object.

    # Returns
        Numpy array containing copy of image data.
    """
    w = img.Width()
    h = img.Height()
    ch = img.Channels()
    out = img.DataCopy()
    out = np.array(out)
    out = np.reshape(out, (h, w, ch))
    out = np.flip(out, axis=2)
    return out

