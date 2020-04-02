import os
import sys
import errno
import numpy as np
import tensorflow as tf
from keras import backend as K

def is_valid_path(parser, arg):
    """ Used as command line argument check for path existence.
        Usage: 
        parser.add_argument("blah", type=lambda x : is_valid_path(parser, x))
        Inputs:
        parser -- Parser being used to parse arguments.
        arg -- Command line argument.
    """
    if not os.path.exists(arg):
        parser.error("Path {0} does not exist!".format(arg))
    else:
        return arg

def ensure_path_exists(path):
    """ Makes sure a path exists.
        Inputs:
        path -- Path to check for existence.
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def constant_length(list_seq, timesteps, num_fea, start=False):
    """ Pads sequence with zeros.
        Inputs:
        list_seq -- List of feature vectors.
        timesteps -- Constant number of timesteps to be enforced.
        num_fea -- Number of features in each feature vector.
        start -- If true, uses beginning of sequence for sequences longer
        than desired length, otherwise uses the end of sequence.
    """
    seq = np.array(list_seq)
    if seq.size == 0:
        return np.zeros((timesteps, num_fea))
    vsize, hsize = seq.shape
    assert hsize == num_fea
    if vsize == timesteps:
        return seq
    elif vsize > timesteps:
        if start:
            return seq[:timesteps, :]
        else:
            return seq[-timesteps:, :]
    else:
        return np.vstack((np.zeros((timesteps - vsize, num_fea)), seq))

def noop(a):
    """ Do nothing function.  Used in augmentation vectors.
    """
    return a

def contrastive_loss(actual, pred):
    """ Contrastive loss function.
        Inputs:
        actual - Actual outputs.
        pred - Predicted outputs.
    """
    margin = 1.0
    return K.mean(
        (1 - actual) * K.square(pred) + 
        actual * K.square(K.maximum(margin - pred, 0)))

def rate(epoch, rate_init, epochs_per_order):
    """ Computes learning rate as a function of epoch index.
        Inputs:
        epoch - Index of current epoch.
        rate_init - Initial rate.
        epochs_per_order - Number of epochs to drop an order of magnitude.
    """
    return rate_init * 10.0 ** (-epoch / epochs_per_order)

def disp(msg):
    """ Prints without crashing on Windows.
        Inputs:
        msg -- Message to print.
    """
    try:
        print(msg)
    except OSError:
        pass

def progress_bar(val, endval, bar_len=30,
        tator=False, tator_msg="", tator_offset=0, tator_span=100, tator_increment=25):
    """ Displays a progress bar.
    """
    if abs(endval) < 0.0000001:
        pct = 1.0
    else:
        pct = float(val) / endval
    pct = min(pct, 1.0)
    if tator:
        if val % tator_increment == 0:
            tator_prog = tator_offset + int(round(pct*tator_span))
            print(f'TATOR_PROGRESS:{tator_prog}:{tator_msg} {val}/{endval}...', flush=True)
    else:
        arrow = '=' * int(round(pct * bar_len) - 1) + '>'
        spaces = '.' * (bar_len - len(arrow))
        sys.stdout.write("\r[{}] {}/{} ({}%)".format(
            arrow + spaces,
            val,
            endval,
            int(round(pct * 100))))

def get_image(track_data_list, index_mapping, index):
    """ Returns an image from a series of track directories.
        Inputs:
        track_data_list -- List of TrackData objects.
        index_mapping -- total num images by 2 array, with each row 
            containing a directory index and the image indices for 
            that directory only.
        index -- Single index, less than total number of images in
            all directories.
    """
    dir_index, orig_index = index_mapping[index]
    return track_data_list[dir_index].detection_image(orig_index)

def get_session(vram_ratio=None):
    """ Sets up tensorflow session with limited memory.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    if vram_ratio:
        config.gpu_options.per_process_gpu_memory_fraction = vram_ratio
    K.tensorflow_backend.set_session(tf.Session(config=config))

def chunks(a, n):
    """ Splits a range into roughly equal sized chunks.
    """
    a_list = list(a)
    k, m = divmod(len(a_list), n)
    return [a_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] 
        for i in list(range(n))]

def parse_paths(config, section, name, check_exists=True):
    """ Parses multiple paths separated by newline.

    # Arguments
        config: ConfigParser object.
        section: Config file section name.
        name: Config file variable name.
    # Returns
        List of paths.
    """
    dirs = config.get(section, name).split("\n")
    dirs = [d for d in dirs if len(d) > 0]
    if check_exists:
        dirs = [d for d in dirs if os.path.exists(d)]
    return dirs
    
