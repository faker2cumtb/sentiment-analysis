# -*- coding: utf-8 -*-

from keras.layers import concatenate
import keras.backend as K
from tensorflow import keras
from utils.ml_utils import MLModel
import tensorflow as tf
# Convolution parameters
kernel_size = [2,3,4,5]
pool_size = 4


class CNN_LSTM_2(MLModel):
    """多核CNN 然后 LSTM 的组合 """
    def __init__(self, hid_dim, dropout_rate, class_dim, **kwargs):
        """Initialize CNN_LSTM model.
        """
        super(CNN_LSTM_2, self).__init__(**kwargs)
        sentence_length = kwargs['sentence_length']
        input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        drop = tf.keras.layers.Dropout(dropout_rate)(self.emb_layer(input_layer))
        conv = []
        for fsz in kernel_size:
            l_conv = tf.keras.layers.Conv1D(hid_dim,
                                fsz,
                                padding='valid',
                                strides=1)(drop)
            BN = tf.keras.layers.BatchNormalization()(l_conv)
            l_conv = tf.keras.layers.Activation(activation='relu')(BN)
            print(l_conv.shape)
            pooling = tf.keras.layers.MaxPooling1D(pool_size=pool_size, padding='valid')(l_conv)
            ##    if strides is None: strides = pool_size
            print("pooling:")
            print(pooling.shape)
            conv.append(pooling)
        merge = tf.keras.layers.concatenate(conv,axis=1)
        # 默认 CNN的filters和LSTM的输出节点数相同
        lstm = tf.keras.layers.LSTM(hid_dim)(merge)
        output = tf.keras.layers.Dense(class_dim, activation='softmax')(lstm)
        model = tf.keras.models.Model(inputs=[input_layer],outputs = output)

        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])
        self.model = model

class LSTM_CNN_2(MLModel):
    """先LSTM 再多核CNN 的组合 """

    def __init__(self, hid_dim, dropout_rate, class_dim, **kwargs):
        """Initialize CNN_LSTM model.
        """
        super(LSTM_CNN_2, self).__init__(**kwargs)
        #model = keras.Sequential()
        sentence_length = kwargs['sentence_length']
        input_layer = tf.keras.layers.Input(shape=(sentence_length,), dtype=tf.int32)
        layer = self.emb_layer(input_layer)
        lstm = tf.keras.layers.LSTM(hid_dim,return_sequences=True)(layer)
        convs = []
        for ksz in kernel_size:

            conv = tf.keras.layers.Conv1D(hid_dim,ksz,padding="valid")(lstm)
            pooling = tf.keras.layers.GlobalMaxPooling1D()(conv)
        # flat = tf.keras.layers.Flatten(pooling)
            convs.append(pooling)
        merge = tf.keras.layers.concatenate(convs,axis=1)

        drop = tf.keras.layers.Dropout(dropout_rate)(merge)
        output = tf.keras.layers.Dense(class_dim, activation="softmax")(drop)

        model = tf.keras.models.Model(inputs=[input_layer],outputs = output)

        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model