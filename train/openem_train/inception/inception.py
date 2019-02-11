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

"""Model definition for inception.
"""

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

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
