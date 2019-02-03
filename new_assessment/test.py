from new_assessment.assess_model import AssessModel
from new_assessment.keras_model_adaptor import KerasModelAdaptor

# K = KerasModelAdaptor('../models/NN00','../data/test_y.npy','../data/test_x.npy',[-1,10000])
# classes = list(range(5))
# A = AssessModel(K,classes)
# A.draw_roc()
# A.metrics()

from new_assessment.hier_assess_model import HierAssessModel
from new_assessment.keras_hier_model_adaptor import KerasHierModelAdaptor

models = ['../models/NN00', '../models/NN10', '../models/NN11', '../models/NN12', '../models/NN13', '../models/NN14']
K = KerasHierModelAdaptor(models_path=models,
                          x_file_path='../data/test_x.npy',
                          y_file_path='../data/test_y.npy',
                          network_size = [1,5],input_shape=[-1,10000])
K.build_network()
A = HierAssessModel(K)
A.hierarchical_assessment(1)