import os
import numpy as np
from keras.optimizers import Adam
from openem_train.ssd import ssd
from openem_train.ssd.ssd_training import MultiboxLoss
from openem_train.ssd.ssd_utils import BBoxUtility
from openem_train.ssd.ssd_dataset import SSDDataset

def train(config):
    """Trains detection model.

    # Arguments
        config: ConfigParser object.
    """

    # Get config file parameters.
    work_dir = config.get('Paths', 'WorkDir')
    num_classes = config.getint('Data', 'NumClasses')
    width = config.getint('Detect', 'Width')
    height = config.getint('Detect', 'Height')

    # Create tensorboard and checkpoints directories.
    check_dir = os.path.join(work_dir, 'detect', 'checkpoints')
    tb_dir = os.path.join(work_dir, 'detect', 'tensorboard')
    os.makedirs(check_dir, exist_ok=True)
    os.makedirs(check_dir, exist_ok=True)

    # Build the ssd model.
    model = ssd.ssd_model((height, width, 3))

    # Set up loss and optimizer.
    loss_obj = MultiboxLoss(
        num_classes,
        neg_pos_ratio=2.0, 
        pos_cost_multiplier=1.1)
    adam = Adam(lr=3e-5)

    # Compile the model.
    model.compile(loss=loss_obj.compute_loss, optimizer=adam)
    model.summary()

    # Get prior box layers from model.
    prior_box_names = [
        'conv4_3_norm_mbox_priorbox',
        'fc7_mbox_priorbox',
        'conv6_2_mbox_priorbox',
        'conv7_2_mbox_priorbox',
        'conv8_2_mbox_priorbox',
        'pool6_mbox_priorbox']
    priors = []
    for prior_box_name in prior_box_names:
        layer = model.get_layer(prior_box_name)
        if layer is not None:
            priors.append(layer.prior_boxes)
    priors = np.vstack(priors)
    
    # Set up bounding box utility.
    bbox_util = BBoxUtility(num_classes, priors)

    # Set up dataset interface.
    dataset = SSDDataset(bbox_util=bbox_util, preprocess_input=lambda x: x)

    
