from Assesment.model_keras_adaptor import ModelKerasAdaptor

keras_model = ModelKerasAdaptor()
keras_model.load('../models/80000NN.h5py')
keras_model.load_data(x_data='../data/test_x.npy',y_data='../data/test_y.npy')
pred_score = keras_model.predict_score(keras_model.X_data)
pred_class = keras_model.predict_classes(keras_model.X_data)
c_metrics = keras_model.metrics(Y_test=keras_model.Y_data,pred_class=pred_class)
