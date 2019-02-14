from keras import models
import numpy as np
from Template.Assessment.model_adaptor import ModelAdaptor

class KerasModelAdaptor(ModelAdaptor):
	"""docstring for SklearnModelAdaptor"""
	def __init__(self, model_file_path, x_test, y_test, shape):
		self.model = models.load_model(model_file_path)
		self.y_file_path = y_test
		self.x_file_path = x_test
		self.shape = shape


	def get_pred_score(self):
		x_test = np.load(self.x_file_path)
		return self.model.predict(x_test)

	
	def get_pred_class(self):
		x_test = np.load(self.x_file_path)
		return self.model.predict_classes(x_test)


	def get_Y(self):
		y_label = np.load(self.y_file_path)
		try:
			return y_label[:,0]
		except:
			return y_label
		