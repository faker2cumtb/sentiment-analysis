# -*- coding: utf-8 -*-

from tensorflow import keras
from utils.ml_utils import MLModel

# Convolution parameters
kernel_size = 5
pool_size = 4


class CNN_LSTM(MLModel):
    """先CNN 然后 LSTM 的组合 """

    def __init__(self, hid_dim, dropout_rate, class_dim, **kwargs):
        """Initialize CNN_LSTM model.
        """
        super(CNN_LSTM, self).__init__(**kwargs)
        model = keras.Sequential()

        model.add(self.emb_layer)
        model.add(keras.layers.Dropout(dropout_rate))
        model.add(keras.layers.Conv1D(hid_dim,
                                      kernel_size,
                                      padding='valid',
                                      activation='relu',
                                      strides=1))
        model.add(keras.layers.MaxPooling1D(pool_size=pool_size))

        # 默认 CNN的filters和LSTM的输出节点数相同
        model.add(keras.layers.LSTM(hid_dim))
        model.add(keras.layers.Dense(class_dim, activation='softmax'))

        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model
