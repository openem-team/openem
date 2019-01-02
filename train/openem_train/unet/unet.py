from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import UpSampling2D
from keras.layers import MaxPooling2D
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.models import Model

def unet_model(input_shape):
    """Constructs U-net model architecture.
    """

    def add_levels(input_tensor, sizes):
        filters = sizes[0]

        down = Conv2D(filters, (3, 3), padding='same')(input_tensor)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)
        down = Conv2D(filters, (3, 3), padding='same')(down)
        down = BatchNormalization()(down)
        down = Activation('relu')(down)

        if len(sizes) == 1:
            return down

        down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down)

        subnet = add_levels(down_pool, sizes[1:])

        up = UpSampling2D((2, 2))(subnet)
        up = concatenate([down, up], axis=3)
        up = Conv2D(filters, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        up = Conv2D(filters, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        up = Conv2D(filters, (3, 3), padding='same')(up)
        up = BatchNormalization()(up)
        up = Activation('relu')(up)
        return up

    inputs = Input(shape=input_shape)
    unet = add_levels(input_tensor=inputs, sizes=[32, 64, 128, 256])
    x = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(unet)

    model = Model(inputs=inputs, outputs=x)
    model.layers[0].name = 'input_1'
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    return model
