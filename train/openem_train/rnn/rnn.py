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

"""Definition of RNN model used for counting.
"""

from keras.layers import Input
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Cropping1D
from keras.layers import Flatten
from keras.layers import Lambda
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

def rnn_model(input_shape, num_steps_crop, unroll=True):
    """Returns RNN model for counting.

    # Arguments
        input_shape: Tuple containing number of timesteps, number of features.
        num_steps_crop: How many timesteps to crop from RNN outputs.

    # Returns
        Keras model object.
    """
    inputs = Input(shape=input_shape, name='input_1')
    out = Bidirectional(GRU(64, return_sequences=True, unroll=unroll))(inputs)
    out = Bidirectional(GRU(16, return_sequences=True, unroll=unroll))(out)
    out = Dense(1, activation='sigmoid')(out)
    out = Cropping1D(cropping=(num_steps_crop, num_steps_crop))(out)
    out = Flatten(name='current_values')(out)
    cumsum1 = Lambda(lambda x: K.cumsum(x))(out)
    cumsum2 = Lambda(lambda x: K.cumsum(K.reverse(x, axes=1)))(out)
    cumsum_value = concatenate([cumsum1, cumsum2], name='cumsum_values')
    model = Model(inputs=inputs, outputs=[out, cumsum_value])
    model.compile(optimizer=Adam(lr=1e-4),
                  loss={'current_values': 'binary_crossentropy', 'cumsum_values': 'mse'},
                  loss_weights={'current_values': 1.0, 'cumsum_values': 0.05})
    return model
