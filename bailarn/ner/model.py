"""
Keras Model
"""
import json
import pickle
from keras.models import Sequential
from keras.layers import Embedding, LSTM, TimeDistributed, Dense, Dropout
from keras.layers.wrappers import Bidirectional
from keras import metrics
from . import constant

from keras_contrib.layers import CRF


class Model(object):
    def __init__(self, embedding_matrix=None):

        self.model_name = 'bi-lstm-crf'

        model = Sequential()

        if embedding_matrix is None:
            model.add(Embedding(constant.WORD_INDEXER_SIZE,
                                constant.EMBEDDING_SIZE, input_length=2000, mask_zero=True))
            model.add(Bidirectional(LSTM(constant.EMBEDDING_SIZE,
                                         dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))

        else:
            model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=2000, weights=[
                      embedding_matrix], mask_zero=True))
            model.add(Bidirectional(LSTM(
                embedding_matrix.shape[1], dropout=0.25, recurrent_dropout=0.25, return_sequences=True)))

        model.add(Dense(constant.NUM_TAGS))

        crf = CRF(constant.NUM_TAGS)

        model.add(crf)
        model.summary()

        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        self.model = model
        print(self.model_name)


def load_model(model_path):
    from keras_contrib.utils import save_load_utils
    model_architecture = Model()
    model = model_architecture.model
    save_load_utils.load_all_weights(
        model, model_path, include_optimizer=False)
    return model


def save_model(model, model_path):
    # from keras_contrib.utils import save_load_utils
    # save_load_utils.save_all_weights(model, model_path)
    model.save(model_path)
