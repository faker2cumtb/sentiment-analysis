"""Model for sentiment analysis.

The model makes use of concatenation of two CNN layers with
different kernel sizes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils.ml_utils import MLModel
from tensorflow import keras

# set parameters:
kernel_size = 3


class CNN_1_layers(MLModel):
    def __init__(self, hid_dim, class_dim, dropout_rate, **kwargs):
        """Initialize CNN model.
		Args:
		  sentence_length: The number of words in each sentence.
			Longer sentences get cut, shorter ones padded.
		  hid_dim: The dimension of the CNN layer.  即filter size
		  class_dim: number of result classes
		  dropout_rate: The portion of dropping value in the Dropout layer.

		"""

        hidden_dims = hid_dim  # dense layer的维度数

        super(CNN_1_layers, self).__init__(**kwargs)
        model = keras.Sequential()

        model.add(self.emb_layer)
        model.add(keras.layers.Conv1D(hid_dim,
                                      kernel_size,
                                      padding='valid',
                                      activation='relu',
                                      strides=1))
        # we use max pooling:
        model.add(keras.layers.GlobalMaxPooling1D())

        # We add a vanilla hidden layer:
        model.add(keras.layers.Dense(hidden_dims))
        model.add(keras.layers.Dropout(dropout_rate))
        model.add(keras.layers.Activation('relu'))

        # We project onto a output layer
        model.add(keras.layers.Dense(class_dim, activation='softmax'))
        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model


class CNN_2_Layers(MLModel):
    """2 CNN layers concat ， 来自google的research model中的例子"""

    def __init__(self, hid_dim, class_dim, dropout_rate, **kwargs):
        """Initialize CNN model.
	
		Args:
		  sentence_length: The number of words in each sentence.
			Longer sentences get cut, shorter ones padded.
		  hid_dim: The dimension of the CNN layer.  即filter size
		  class_dim: number of result classes
		  dropout_rate: The portion of dropping value in the Dropout layer.

		"""
        super(CNN_2_Layers, self).__init__(**kwargs)

        sentence_length = kwargs['sentence_length']
        input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)

        layer = self.emb_layer(input_layer)

        layer_conv3 = tf.keras.layers.Conv1D(hid_dim, kernel_size, activation="relu")(layer)
        layer_conv3 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv3)
        print(layer_conv3)

        layer_conv4 = tf.keras.layers.Conv1D(hid_dim, kernel_size - 1, activation="relu", )(layer)
        layer_conv4 = tf.keras.layers.GlobalMaxPooling1D()(layer_conv4)
        print(layer_conv3)
        layer = tf.keras.layers.concatenate([layer_conv4, layer_conv3], axis=1)
        print(layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        layer = tf.keras.layers.Dropout(dropout_rate)(layer)

        output = tf.keras.layers.Dense(class_dim, activation="softmax")(layer)

        model = tf.keras.models.Model(inputs=[input_layer], outputs=output)
        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model
