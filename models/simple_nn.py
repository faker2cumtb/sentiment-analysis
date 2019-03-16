# -*- coding: utf-8 -*-

from tensorflow import keras
import tensorflow as tf
from utils.ml_utils import MLModel


class SimpleNN(MLModel):
    """one hidden layer for sentimental analysis.  来自google的tutorial"""

    def __init__(self, hid_dim, class_dim, **kwgs):
        """Initialize simple NN model.
		"""
        super(SimpleNN, self).__init__(**kwgs)
        model = keras.Sequential()

        model.add(self.emb_layer)
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(hid_dim, activation=tf.nn.relu))
        model.add(keras.layers.Dense(class_dim, activation=tf.nn.softmax))

        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model
