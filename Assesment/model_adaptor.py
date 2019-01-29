from abc import abstractmethod


class IModelAdaptor:
    def __init__(self):
        self.classifier_model = None

    @abstractmethod
    def load(self, model_file_path):
        pass

    @abstractmethod
    def file_postfix(self):
        pass

    @abstractmethod
    def serialize(self, out_obj):
        pass

    @abstractmethod
    def deserialize(self, in_str, **kwargs):
        pass

    @abstractmethod
    def predict_score(self, in_obj):
        pass

    @abstractmethod
    def predict_classes(self, in_obj):
        pass

    @abstractmethod
    def draw_RoC(self, **kwargs):
        pass

    @abstractmethod
    def metrics(self, **kwargs):
        pass

    @abstractmethod
    def load_data(self, **kwargs):
        pass

