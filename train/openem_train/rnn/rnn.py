from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Cropping1D
from keras.layers import Flatten
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

def rnn_model(input_shape, num_steps_crop, unroll=True):
    inputs = Input(shape=input_shape)
    x = Bidirectional(GRU(64, return_sequences=True, unroll=unroll))(inputs)
    x = Bidirectional(GRU(16, return_sequences=True, unroll=unroll))(x)
    x = Dense(1, activation='sigmoid')(x)
    x = Cropping1D(cropping=(num_steps_crop, num_steps_crop))(x)
    x = Flatten(name='current_values')(x)
    cumsum_value = Lambda(lambda a: K.cumsum(a, axis=1), name='cumsum_values')(x)
    model = Model(inputs=inputs, outputs=[x, cumsum_value])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'current_values': 'binary_crossentropy', 'cumsum_values': 'mse'},
                  loss_weights={'current_values': 1.0, 'cumsum_values': 0.001})
    return model
