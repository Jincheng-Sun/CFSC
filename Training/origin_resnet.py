import pandas as pd
import numpy as np
from keras import models, Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, GlobalAveragePooling2D, add, Input, Dense, \
    MaxPool2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras import regularizers
from keras import optimizers
from keras.callbacks import EarlyStopping


def bn_relu(layer, dropout=0, **params):
    layer = BatchNormalization()(layer)
    layer = Activation(params['conv_activation'])(layer)

    if dropout > 0:
        layer = Dropout(dropout)(layer)
    return layer

def global_average_pooling(layer, cls):
    layer = Conv2D(cls, [1, 1])(layer)
    layer = GlobalAveragePooling2D()(layer)
    layer = Activation(activation='softmax')(layer)
    return layer

def ResBlock_type1(layer, filters, kernels, dropout, activation, shift=False, shrink=False):
    # -Conv-BN-Act-Conv-BN-Act-
    # ↳-----Conv-BN-------↑

    shape = [1, 1]
    if shrink:
        shape = [2, 2]

    if shift:
        shortcut = Conv2D(filters=filters,
                          kernel_size=shape,
                          kernel_initializer='random_uniform',
                          # kernel_regularizer=regularizers.l2(0.01),
                          strides=shape,
                          padding='same')(layer)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = layer

    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=shape,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filters,
                   kernel_size=kernels,
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=[1, 1],
                   padding='same')(layer)

    layer = BatchNormalization()(layer)

    layer = add([shortcut, layer])

    layer = Activation(activation=activation)(layer)

    return layer


def ResBlock_type2(layer, filters, kernels, activation, dropout = 0, shift=False, shrink=False):
    # -Conv-BN-Act-Conv-BN-Act-conv-BN-Act-
    # ↳-------------Conv-BN------------↑
    filter1, filter2, filter3 = filters
    kernel1, kernel2, kernel3 = kernels
    shape = [1, 1]
    if shrink:
        shape = [2, 2]

    if shift:
        shortcut = Conv2D(filters=filter3,
                          kernel_size=[kernel1, kernel1],
                          kernel_initializer='random_uniform',
                          # kernel_regularizer=regularizers.l2(0.01),
                          strides=shape,
                          padding='same')(layer)
        shortcut = BatchNormalization()(shortcut)
    else:
        shortcut = layer
    layer = Conv2D(filters=filter1,
                   kernel_size=[kernel1, kernel1],
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=shape,
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter2,
                   kernel_size=[kernel2, kernel2],
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=[1, 1],
                   padding='same')(layer)
    layer = bn_relu(layer, dropout=dropout, conv_activation=activation)

    layer = Conv2D(filters=filter3,
                   kernel_size=[kernel3, kernel3],
                   kernel_initializer='random_uniform',
                   # kernel_regularizer=regularizers.l2(0.01),
                   strides=[1, 1],
                   padding='same')(layer)

    layer = BatchNormalization()(layer)

    output = add([shortcut, layer])

    output = Activation(activation=activation)(output)

    return output
