from sklearn.externals import joblib


class SklearnModelAdaptor(ModelAdaptor):
	"""docstring for SklearnModelAdaptor"""
	def __init__(self, onvsrest_model_file_path, model_file_path, y_file_path, x_file_path):
		self.onvsrest_model_file_path = onvsrest_model_file_path
		self.model_file_path = model_file_path
		self.y_file_path = y_file_path
		self.x_file_path = x_file_path


	def get_pred_score():
		"""
		classifiers = OneVsRestClassifier(BernoulliNB(alpha=1, fit_prior=True))
		Y_score = classifiers.fit(X_train, Y_train).predict_proba(X_test)
		"""
		X_test = _get_X(self)
		model = joblib.load(self.onvsrest_model_file_path)
		return model.predict_proba(X_test)
	
	
	def get_pred_class(self):
		X_test = _get_X()
		model = joblib.load(self.model_file_path)
		return model.predict(X_test)


	def get_Y():
		