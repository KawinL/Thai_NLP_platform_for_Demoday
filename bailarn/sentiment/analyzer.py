import csv
import gc
import glob
import json
import os
import shutil
import sys
import warnings
from collections import Counter

import pickle  # json

import numpy as np
import pandas as pd

from sklearn.exceptions import UndefinedMetricWarning
import sklearn.metrics

from . import constant
from .model import load_model, save_model, Model
from .metric import custom_metric
from .callback import CustomCallback
from ..utils import utils

from types import SimpleNamespace

import tensorflow as tf


class SentimentAnalyzer(object):

    def __init__(self, model_path=None, new_model=False, tag_index=None, embedding_matrix=None):

        self.new_model = new_model
        self.model_path = model_path
        self.tag_index = tag_index
        if self.tag_index is None:
            self.tag_index = utils.build_tag_index(
                constant.TAG_LIST, start_index=0)

        self.embedding_matrix = embedding_matrix
        if self.new_model is True:
            if self.model_path is None:
                self.model = Model(
                    embedding_matrix=self.embedding_matrix).model

            else:
                raise ValueError('model_path must be none for new model')
        else:
            if self.model_path is None:
                self.model = load_model(os.path.join(os.path.dirname(
                    os.path.realpath(__file__)), constant.DEFAULT_MODEL_PATH))
            else:
                if not os.path.exists(model_path):
                    raise ValueError("File " + model_path + " does not exist")
                self.model = load_model(model_path)
        
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def evaluate(self, x_true, y_true):
        
        with self.graph.as_default():
            pred = self.model.predict(x_true)

        amax_pred = np.argmax(pred, axis=1)
        amax_true = np.argmax(y_true, axis=1)
        pred_flat = amax_pred.flatten()
        true_flat = amax_true.flatten()

        scores = custom_metric(true_flat, pred_flat)

        for score in scores:
            print(score, ": ", scores[score])
        return scores

    def train(self, x_train, y_train, train_name='untitled', validation_split=0.1, epochs=1, batch_size=32):
        callbacks = CustomCallback(train_name, validation_split).callbacks
        self.model.fit(x_train, y_train, validation_split=validation_split,
                       epochs=epochs, batch_size=batch_size, shuffle=shuffle, callbacks=callbacks)

        self.new_model = False

    def save(self, model_path):
        save_model(self.model, model_path)

    def predict(self, x, decode_tag=True):
        inv_map = {v: k for k, v in self.tag_index.items()}
        
        with self.graph.as_default():
            pred = self.model.predict(x)

        if decode_tag:
            pred = np.argmax(pred, axis=1)
            pred = np.array([inv_map[p] for p in pred])

        return pred
