import pandas as pd
import numpy as np
from keras import models, Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils


model = Sequential()
model.add(Conv2D(256, input_shape=[100,100,1],kernel_size=[2, 100], strides=[1,100], padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()