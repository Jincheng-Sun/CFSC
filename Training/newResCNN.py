import pandas as pd
import numpy as np
from keras import models, Sequential
from keras.layers import BatchNormalization, Conv2D, Activation, Dropout, GlobalAveragePooling2D, add, Input, Dense,Flatten
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix
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

def res_block(input,filters,stride,dim_up = False):

    if dim_up:
        shortcut = Conv2D(filters = filters,
                          kernel_size=[1,1],
                          use_bias=False)(input)
    else:
        shortcut = input

    layer = Conv2D(filters=filters,
                   kernel_size=[1,stride],
                   kernel_initializer='random_uniform',
                   padding='same')(input)
    layer = bn_relu(layer,conv_activation='relu')
    layer = Conv2D(filters=filters,
                   kernel_size=[1,stride],
                   kernel_initializer='random_uniform',
                   padding='same')(layer)
    layer = add([layer,shortcut])
    layer = bn_relu(layer, conv_activation='relu')
    return layer

def network(num):
    input = Input(shape=[100, 100, 1])
    layer = Conv2D(filters=64,kernel_size=[1,100],strides=[1,100],kernel_initializer='random_uniform')(input)
    layer = bn_relu(layer,conv_activation='relu')
    layer = res_block(layer,64,stride=2)
    layer = res_block(layer, 128, stride=4, dim_up=True)
    layer = res_block(layer, 256, stride=8, dim_up=True)
    layer = res_block(layer, 512, stride=16, dim_up=True)
    layer = Flatten()(layer)
    output = Dense(num,activation='softmax')(layer)
    model = Model(inputs=[input], outputs=[output])
    model.summary()

    return model

def train(model):
    num = 19998
    X_train = np.load('../data/train_x.npy')[0:num]
    Y_train = np.load('../data/train_y.npy')[0:num]
    X_train = np.reshape(X_train, [num, 100, 100, 1])
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
    y_train = pd.DataFrame(y_train)[0]
    y_val = pd.DataFrame(y_val)[0]
    # one-hot，5 category
    y_labels = list(y_train.value_counts().index)
    # y_labels = np.unique(y_train)
    le = preprocessing.LabelEncoder()
    le.fit(y_labels)
    num_labels = len(y_labels)
    y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
    y_val = to_categorical(y_val.map(lambda x: le.transform([x])[0]), num_labels)

    adam = optimizers.adam(lr=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

    model.fit(x_train, y_train,
              batch_size=50,
              epochs=20,
              validation_data=(x_val, y_val),
              callbacks=[monitor])
    score = model.evaluate(x_val, y_val, verbose=0)
    val_loss = score[0]
    acc = score[1]
    model.save('../models/RCNN')
    X_test = np.load('../data/test_x.npy')
    Y_test = np.load('../data/test_y.npy')
    # X_train reshape to [40000,100,100]
    X_test = np.reshape(X_test, [4000, 100, 100, 1])
    score = model.predict(X_test)
    score = np.argmax(score, axis=1)
    score = accuracy_score(score, Y_test)

    print(score)

model = network(5)
train(model)
