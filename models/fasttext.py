# -*- coding: utf-8 -*-

from utils.ml_utils import MLModel
from tensorflow import keras

# Set parameters:
# ngram_range = 2 will add bi-grams features
# ngram_range = 1
# max_features = 20000
# maxlen = 400
# batch_size = 32
# embedding_dims = 50
# epochs = 5


class FastText(MLModel):

    def __init__(self, class_dim, **kwargs):
        """Initialize CNN_LSTM model.
        """
        super(FastText, self).__init__(**kwargs)
        model = keras.Sequential()

        model.add(self.emb_layer)
        # we start off with an efficient embedding layer which maps
        # our vocab indices into embedding_dims dimensions
        # model.add(Embedding(max_features,
        #                     embedding_dims,
        #                     input_length=maxlen))

        # we add a GlobalAveragePooling1D, which will average the embeddings
        # of all words in the document
        model.add(keras.layers.GlobalAveragePooling1D())

        # We project onto a single unit output layer, and squash it with a sigmoid:
        model.add(keras.layers.Dense(class_dim, activation='softmax'))
        model.compile(loss="categorical_crossentropy",  # "binary_crossentropy"
                      optimizer="adam",  # "adam"  rmsprop
                      metrics=["accuracy"])

        self.model = model
