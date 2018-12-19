"""Model definition for inception.
"""

from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.optimizers import SGD

def inception_model(input_shape, num_classes):
    """Builds inceptionv3 network.

    # Arguments
        input_shape: Shape of input tensor.
        num_classes: Number of classification classes.
    """
    img_input = Input(input_shape, name='data')
    base_model = InceptionV3(
        input_tensor=img_input,
        include_top=False,
        pooling='avg')
    species_dense = Dense(
        num_classes,
        activation='softmax',
        name='cat_species')(base_model.layers[-1].output)
    cover_dense = Dense(
        3,
        activation='softmax',
        name='cat_cover')(base_model.layers[-1].output)
    model = Model(input=img_input, outputs=[species_dense, cover_dense])
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
        optimizer=sgd,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
