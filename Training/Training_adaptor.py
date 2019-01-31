from abc import abstractmethod


class Training_adaptor:

    def __init__(self):
        pass

    @abstractmethod
    def load_data(self, x_train, y_train, x_test, y_test, **kwargs):
        pass

    @abstractmethod
    def create_dataset(self, **kwargs):
        pass

    @abstractmethod
    def network(self, **kwargs):
        pass

    @abstractmethod
    def load_model(self, model):
        pass

    @abstractmethod
    def train_model(self, model, X_train, Y_train, **kwargs):
        pass

    @abstractmethod
    def assesment(self, model, X_test, Y_test):
        pass
