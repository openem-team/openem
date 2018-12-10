"""Functions for defining SSD model architecture.
"""

import keras.backend as K
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import GlobalAveragePooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import Reshape
from keras.layers import ZeroPadding2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.applications.resnet50 import ResNet50

from openem_train.ssd.ssd_layers import Normalize
from openem_train.ssd.ssd_layers import PriorBox

def _wrap_with_bn(net, input_tensor, name):
    net['bn'+name] = BatchNormalization(name='bn'+name)(input_tensor)
    net['act'+name] = Activation('relu')(net['bn'+name])
    return net['act'+name]

def _wrap_with_bn_and_dropout(net, input_tensor, name):
    net['dropout' + name] = Dropout(0.25)(_wrap_with_bn(net, input_tensor, name))
    return net['dropout' + name]

def ssd_model(input_shape, num_classes=21):
    """SSD300 + Resnet50 architecture.

    # Arguments
        input_shape: Shape of the input image, expected to be (300, 300, 3).
        num_classes: Number of classes including background.

    # References
        https://arxiv.org/abs/1512.02325
    """
    net = {}
    img_size = (input_shape[1], input_shape[0])
    input_tensor = Input(shape=input_shape)
    net['input'] = input_tensor

    resnet_head = ResNet50(include_top=False, input_tensor=input_tensor)

    activation_layer_names = []

    for layer in resnet_head.layers:
        if layer._outbound_nodes:
            net[layer.name] = layer.output
            layer.trainable = False
            if layer.name.startswith('activation'):
                activation_layer_names.append(layer.name)

    # The ResNet50 architecture changed in June 2018 to exclude
    # an average pooling layer when include_top is false. This caused
    # the last activation layer to no longer have output nodes, so
    # we add it to the list of activation layers here if it is missing.
    if 'activation_49' not in activation_layer_names:
        net['activation_49'] = resnet_head.layers[-1].output
        resnet_head.layers[-1].trainable = False
        activation_layer_names.append('activation_49')

    prev_size_layer_name = activation_layer_names[-10]

    net['pool5'] = MaxPooling2D(
        (3, 3), strides=(1, 1), padding='same',
        name='pool5')(net[activation_layer_names[-1]])
    # FC6
    net['fc6'] = Conv2D(
        1024, (3, 3), dilation_rate=(6, 6),
        activation='relu', padding='same',
        name='fc6')(net['pool5'])
    # FC7
    net['fc7'] = Conv2D(
        1024, (1, 1), activation='relu',
        padding='same', name='fc7')(net['fc6'])

    # Block 6
    net['conv6_1'] = _wrap_with_bn_and_dropout(
        net,
        Conv2D(256, (1, 1), activation='linear',
               padding='same',
               name='conv6_1')(net['fc7']),
        name='6_1')

    net['conv6_2'] = Conv2D(
        512, (3, 3), strides=(2, 2),
        activation='relu', padding='same',
        name='conv6_2')(net['conv6_1'])
    # Block 7
    net['conv7_1'] = _wrap_with_bn_and_dropout(
        net,
        Conv2D(128, (1, 1), activation='linear',
               padding='same',
               name='conv7_1')(net['conv6_2']),
        name='7_1')

    net['conv7_2'] = ZeroPadding2D()(net['conv7_1'])
    net['conv7_2'] = Conv2D(
        256, (3, 3), strides=(2, 2),
        activation='relu', padding='valid',
        name='conv7_2')(net['conv7_2'])
    # Block 8
    net['conv8_1'] = _wrap_with_bn_and_dropout(
        net,
        Conv2D(128, (1, 1), activation='relu',
               padding='same',
               name='conv8_1')(net['conv7_2']),
        name='8_1')

    net['conv8_2'] = Conv2D(
        256, (3, 3), strides=(2, 2),
        activation='relu', padding='same',
        name='conv8_2')(net['conv8_1'])
    # Last Pool
    net['pool6'] = GlobalAveragePooling2D(name='pool6')(net['conv8_2'])
    # Prediction from conv4_3
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm')(net[prev_size_layer_name])
    num_priors = 3
    x_out = Conv2D(
        num_priors * 4, (3, 3), padding='same',
        name='conv4_3_norm_mbox_loc')(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc'] = x_out
    flatten = Flatten(name='conv4_3_norm_mbox_loc_flat')
    net['conv4_3_norm_mbox_loc_flat'] = flatten(net['conv4_3_norm_mbox_loc'])
    name = 'conv4_3_norm_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x_out = Conv2D(
        num_priors * num_classes, (3, 3), padding='same',
        name=name)(net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf'] = x_out
    flatten = Flatten(name='conv4_3_norm_mbox_conf_flat')
    net['conv4_3_norm_mbox_conf_flat'] = flatten(net['conv4_3_norm_mbox_conf'])
    priorbox = PriorBox(
        img_size, 30.0, aspect_ratios=[2],
        variances=[0.1, 0.1, 0.2, 0.2],
        name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox(net['conv4_3_norm'])
    # Prediction from fc7
    num_priors = 6
    net['fc7_mbox_loc'] = Conv2D(
        num_priors * 4, (3, 3),
        padding='same',
        name='fc7_mbox_loc')(net['fc7'])
    flatten = Flatten(name='fc7_mbox_loc_flat')
    net['fc7_mbox_loc_flat'] = flatten(net['fc7_mbox_loc'])
    name = 'fc7_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    net['fc7_mbox_conf'] = Conv2D(
        num_priors * num_classes, (3, 3),
        padding='same',
        name=name)(net['fc7'])
    flatten = Flatten(name='fc7_mbox_conf_flat')
    net['fc7_mbox_conf_flat'] = flatten(net['fc7_mbox_conf'])
    priorbox = PriorBox(
        img_size, 60.0, max_size=114.0, aspect_ratios=[2, 3],
        variances=[0.1, 0.1, 0.2, 0.2],
        name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])
    # Prediction from conv6_2
    num_priors = 6
    x_out = Conv2D(
        num_priors * 4, (3, 3), padding='same',
        name='conv6_2_mbox_loc')(net['conv6_2'])
    net['conv6_2_mbox_loc'] = x_out
    flatten = Flatten(name='conv6_2_mbox_loc_flat')
    net['conv6_2_mbox_loc_flat'] = flatten(net['conv6_2_mbox_loc'])
    name = 'conv6_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x_out = Conv2D(
        num_priors * num_classes, (3, 3), padding='same',
        name=name)(net['conv6_2'])
    net['conv6_2_mbox_conf'] = x_out
    flatten = Flatten(name='conv6_2_mbox_conf_flat')
    net['conv6_2_mbox_conf_flat'] = flatten(net['conv6_2_mbox_conf'])
    priorbox = PriorBox(
        img_size, 114.0, max_size=168.0, aspect_ratios=[2, 3],
        variances=[0.1, 0.1, 0.2, 0.2],
        name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])
    # Prediction from conv7_2
    num_priors = 6
    x_out = Conv2D(
        num_priors * 4, (3, 3), padding='same',
        name='conv7_2_mbox_loc')(net['conv7_2'])
    net['conv7_2_mbox_loc'] = x_out
    flatten = Flatten(name='conv7_2_mbox_loc_flat')
    net['conv7_2_mbox_loc_flat'] = flatten(net['conv7_2_mbox_loc'])
    name = 'conv7_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x_out = Conv2D(
        num_priors * num_classes, (3, 3), padding='same',
        name=name)(net['conv7_2'])
    net['conv7_2_mbox_conf'] = x_out
    flatten = Flatten(name='conv7_2_mbox_conf_flat')
    net['conv7_2_mbox_conf_flat'] = flatten(net['conv7_2_mbox_conf'])
    priorbox = PriorBox(
        img_size, 168.0, max_size=222.0, aspect_ratios=[2, 3],
        variances=[0.1, 0.1, 0.2, 0.2],
        name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])
    # Prediction from conv8_2
    num_priors = 6
    x_out = Conv2D(
        num_priors * 4, (3, 3), padding='same',
        name='conv8_2_mbox_loc')(net['conv8_2'])
    net['conv8_2_mbox_loc'] = x_out
    flatten = Flatten(name='conv8_2_mbox_loc_flat')
    net['conv8_2_mbox_loc_flat'] = flatten(net['conv8_2_mbox_loc'])
    name = 'conv8_2_mbox_conf'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x_out = Conv2D(
        num_priors * num_classes, (3, 3), padding='same',
        name=name)(net['conv8_2'])
    net['conv8_2_mbox_conf'] = x_out
    flatten = Flatten(name='conv8_2_mbox_conf_flat')
    net['conv8_2_mbox_conf_flat'] = flatten(net['conv8_2_mbox_conf'])
    priorbox = PriorBox(
        img_size, 222.0, max_size=276.0, aspect_ratios=[2, 3],
        variances=[0.1, 0.1, 0.2, 0.2],
        name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])
    # Prediction from pool6
    num_priors = 6
    x_out = Dense(num_priors * 4, name='pool6_mbox_loc_flat')(net['pool6'])
    net['pool6_mbox_loc_flat'] = x_out
    name = 'pool6_mbox_conf_flat'
    if num_classes != 21:
        name += '_{}'.format(num_classes)
    x_out = Dense(num_priors * num_classes, name=name)(net['pool6'])
    net['pool6_mbox_conf_flat'] = x_out
    priorbox = PriorBox(
        img_size, 276.0, max_size=330.0, aspect_ratios=[2, 3],
        variances=[0.1, 0.1, 0.2, 0.2],
        name='pool6_mbox_priorbox')
    if K.image_dim_ordering() == 'tf':
        target_shape = (1, 1, 256)
    else:
        target_shape = (256, 1, 1)
    net['pool6_reshaped'] = Reshape(
        target_shape,
        name='pool6_reshaped')(net['pool6'])
    net['pool6_mbox_priorbox'] = priorbox(net['pool6_reshaped'])
    # Gather all predictions
    net['mbox_loc'] = concatenate(
        [net['conv4_3_norm_mbox_loc_flat'],
         net['fc7_mbox_loc_flat'],
         net['conv6_2_mbox_loc_flat'],
         net['conv7_2_mbox_loc_flat'],
         net['conv8_2_mbox_loc_flat'],
         net['pool6_mbox_loc_flat']],
        axis=1, name='mbox_loc')
    net['mbox_conf'] = concatenate(
        [net['conv4_3_norm_mbox_conf_flat'],
         net['fc7_mbox_conf_flat'],
         net['conv6_2_mbox_conf_flat'],
         net['conv7_2_mbox_conf_flat'],
         net['conv8_2_mbox_conf_flat'],
         net['pool6_mbox_conf_flat']],
        axis=1, name='mbox_conf')
    net['mbox_priorbox'] = concatenate(
        [net['conv4_3_norm_mbox_priorbox'],
         net['fc7_mbox_priorbox'],
         net['conv6_2_mbox_priorbox'],
         net['conv7_2_mbox_priorbox'],
         net['conv8_2_mbox_priorbox'],
         net['pool6_mbox_priorbox']],
        axis=1, name='mbox_priorbox')
    if hasattr(net['mbox_loc'], '_keras_shape'):
        num_boxes = net['mbox_loc']._keras_shape[-1] // 4
    elif hasattr(net['mbox_loc'], 'int_shape'):
        num_boxes = K.int_shape(net['mbox_loc'])[-1] // 4
    net['mbox_loc'] = Reshape(
        (num_boxes, 4),
        name='mbox_loc_final')(net['mbox_loc'])
    net['mbox_conf'] = Reshape(
        (num_boxes, num_classes),
        name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation(
        'softmax',
        name='mbox_conf_final')(net['mbox_conf'])
    net['predictions'] = concatenate(
        [net['mbox_loc'],
         net['mbox_conf'],
         net['mbox_priorbox']],
        axis=2,
        name='predictions')
    model = Model(net['input'], net['predictions'])
    model.layers[0].name = 'input_1'
    return model
