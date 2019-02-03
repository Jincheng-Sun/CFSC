from keras import models
import numpy as np
from new_assessment.model_adaptor import ModelAdaptor

class KerasModelAdaptor(ModelAdaptor):
	"""docstring for SklearnModelAdaptor"""
	def __init__(self, model_file_path, y_file_path, x_file_path, shape):
		self.model = models.load_model(model_file_path)
		self.y_file_path = y_file_path
		self.x_file_path = x_file_path
		self.shape = shape


	def get_pred_score(self):
		x_test = np.reshape(np.load(self.x_file_path), newshape=self.shape)
		return self.model.predict(x_test)

	
	def get_pred_class(self):
		x_test = np.reshape(np.load(self.x_file_path), newshape=self.shape)
		return self.model.predict_classes(x_test)


	def get_Y(self):
		first_layer_label = np.load(self.y_file_path)
		first_layer_label = first_layer_label[:,0]
		return first_layer_label
		