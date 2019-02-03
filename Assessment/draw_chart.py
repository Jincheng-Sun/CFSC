import sys

sys.path.append('/home/oem/Projects/CFSC')
from Assessment.model_keras_adaptor import ModelKerasAdaptor
import matplotlib.pyplot as plt
import numpy as np

keras_model = ModelKerasAdaptor()
keras_model.load('../models/NN00')
keras_model.load_data(x_data='../data/test_x.npy', y_data='../data/test_y.npy')
pred_score = keras_model.predict_score(keras_model.X_data)
pred_class = keras_model.predict_classes(keras_model.X_data)
confusion = keras_model.metrics(Y_test=keras_model.Y_data, pred_class=pred_class)

name_list = ['real class 0', 'real class 1', 'real class 2', 'real class 3', 'real class 4']

x = np.arange(len(name_list))
total_width, n = 0.8, 5
width = total_width / n
# x = x - (total_width - width) / 2

plt.bar(x, confusion[:, 0], label='predict class 0', width=width)
plt.bar(x + width, confusion[:, 1], label='predict class 1', width=width)
plt.bar(x + width * 2, confusion[:, 2], label='predict class 2', tick_label=name_list, width=width)
plt.bar(x + width * 3, confusion[:, 3], label='predict class 3', width=width)
plt.bar(x + width * 4, confusion[:, 4], label='predict class 4', width=width)

# plt.
plt.legend()
plt.show()
