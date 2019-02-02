from abc import abstractmethod


class Hierarchical_Adaptor:
    def __init__(self):
        self.model_type = None
        self.network_size = None

    @abstractmethod
    def init_params(self, model_type, network_size):
        pass

    @abstractmethod
    def load_data(self, x_file_path, y_file_path, models_path):
        pass

    @abstractmethod
    def build_network(self):
        pass

    @abstractmethod
    def fit_data(self, models_all):
        pass

    @abstractmethod
    def hierarchical_acc(self, pred_list, real_list, n):
        pass
