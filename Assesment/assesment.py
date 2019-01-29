from Assesment.model_keras_adaptor import ModelKerasAdaptor
from Assesment.model_sklearn_adaptor import ModelSklearnAdaptor
import numpy as np

class Assesment:
	def __init__(self):
		self.model = None
		self.X_test = None
		self.Y_test = None


	def load_model(self, model_file_path):
		if model_file_path.endswith('.joblib'):
			self.model = ModelSklearnAdaptor()
		else:
			self.model = ModelKerasAdaptor()

		self.model.load(model_file_path)


	def load_data(self, x_filepath, y_filepath, input_shape):
		self.X_test, self.Y_test = self.model.deserialize(x_filepath=x_filepath, y_filepath=y_filepath, shape=input_shape)


	def assess(self):
		if self.model != None and self.X_test is not None and self.Y_test is not None:
			pred_score = self.model.predict_score(in_obj=self.X_test)
			pred_class = self.model.predict_classes(in_obj=self.X_test)
			self.model.draw_RoC(Y_test=self.Y_test,pred_score=pred_score)
			self.model.metrics(Y_test=self.Y_test,pred_class=pred_class)
		else:
			print("[INFO] model and data missing")



asses = Assesment()
asses.load_model('../models/80000NN.h5py')
asses.load_data(x_filepath='../data/test_x.npy',y_filepath='../data/test_y.npy',input_shape=[4000,10000])
asses.assess()