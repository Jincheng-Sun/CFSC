from abc import abstractmethod


class ModelAdaptor():
	"""docstring for ModelAdaptor"""
	def __init__(self):


	@abstractmethod
	def get_Y():
		pass

	@abstractmethod
	def get_pred_score():
		pass

	@abstractmethod
	def get_pred_class():
		pass