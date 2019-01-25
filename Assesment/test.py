from Assesment.model_keras_adaptor import ModelKerasAdaptor

kerasmodel = ModelKerasAdaptor()
# kerasmodel.load('../models/80000NN')
#
# model_type = kerasmodel.file_postfix()
#
X_test, Y_test = kerasmodel.deserialize(x_filepath='../data/test_x.npy',
                           y_filepath='../data/test_y.npy',
                           shape=[4000,10000])
#
# pred_score = kerasmodel.predict_score(in_obj=X_test)
# pred_class = kerasmodel.predict_classes(in_obj=X_test)
#
# kerasmodel.draw_RoC(Y_test=Y_test,pred_score=pred_score)
# kerasmodel.metrics(Y_test=Y_test,pred_class=pred_class)
#
# json = kerasmodel.serialize(pred_class)

import json

jsonData = '{"a":1,"b":2,"c":3,"d":4,"e":5}';

text = json.loads(jsonData)

x,y=kerasmodel.convert_data(text)