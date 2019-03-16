# -*- coding: utf-8 -*-

from tensorflow import keras
# import keras
from utils.ml_utils import MLModel


class SimpleLSTM(MLModel):
    """LSTM for sentimental analysis."""

    def __init__(self, hid_dim, dropout_rate, class_dim, **kwargs):
        """Initialize LSTM model.
        """
        super(SimpleLSTM, self).__init__(**kwargs)
        model = keras.Sequential()

        model.add(self.emb_layer)
        model.add(keras.layers.LSTM(hid_dim, dropout=dropout_rate,
                                    recurrent_dropout=dropout_rate))
        model.add(keras.layers.Dense(class_dim, activation='softmax'))
        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model


class BiLSTM(MLModel):
    """Bi LSTM for sentimental analysis."""

    def __init__(self, hid_dim, class_dim, dropout_rate, **kwargs):
        """Initialize Bi-LSTM model.
        """

        super(BiLSTM, self).__init__(**kwargs)
        model = keras.Sequential()

        model.add(self.emb_layer)
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(hid_dim)))
        model.add(keras.layers.Dropout(dropout_rate))
        model.add(keras.layers.Dense(class_dim, activation='softmax'))
        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model


class StackedLSTM(MLModel):
    """两层LSTM 堆叠 参考腾讯公众号文章"""

    def __init__(self, hid_dim, class_dim, dropout_rate, **kwargs):
        """Initialize Bi-LSTM model.
        """

        super(StackedLSTM, self).__init__(**kwargs)
        model = keras.Sequential()
        hid_dim = 64
        dropout_rate = 0.1

        model.add(self.emb_layer)
        # l = keras.layers.Embedding(10000, 128, input_length=40)
        # model.add(l)
        model.add(keras.layers.LSTM(hid_dim, dropout=dropout_rate, return_sequences=True))
        model.add(keras.layers.LSTM(hid_dim, return_sequences=True))
        # inputs = keras.layers.Input(shape=(3, 2, 4))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(class_dim))  # , activation = 'softmax'))
        model.add(keras.layers.Activation('softmax'))
        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model


class GRU(MLModel):
    """LSTM for sentimental analysis."""

    def __init__(self, hid_dim, dropout_rate, class_dim, **kwargs):
        """Initialize LSTM model.
        """
        super(GRU, self).__init__(**kwargs)
        model = keras.Sequential()

        model.add(self.emb_layer)
        model.add(keras.layers.GRU(hid_dim, dropout=dropout_rate,
                                   recurrent_dropout=dropout_rate))
        model.add(keras.layers.Dense(class_dim, activation='softmax'))
        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model
