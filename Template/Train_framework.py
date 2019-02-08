from Template.trainKerasModel import TrainKerasModel
from .Residual_Network import Res50

'''Initialize file path'''

X_train =
Y_train =
X_val =
Y_val =
X_test =
Y_test =
save_path =

'''Initialize training settings'''

epoch =
batch_size =
input_shape =
output_classes =
# optional, modify if you like
from keras.callbacks import EarlyStopping

call_backs = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
loss = 'categorical_crossentropy'
optimizer = 'adam'

'''generalize network'''

model = Res50(output_classes, input_shape)

'''train model'''
train = TrainKerasModel(model=model,
                        X_train=X_train, Y_train=Y_train,
                        X_val=X_val, Y_val=Y_val)
train.train(batch_size=batch_size, epoch=epoch,
            call_backs=call_backs, loss=loss, optimizer=optimizer)
model = train.model
train.save(save_path)

'''assessment'''

from .Assessment.keras_model_adaptor import KerasModelAdaptor
from .Assessment.assess_model import AssessModel

AssessKeras = KerasModelAdaptor(model_file_path=save_path,
                                y_file_path=Y_test, x_file_path=X_test,
                                shape=input_shape)
Assessment = AssessModel(AssessKeras)
Assessment.draw_roc()
Assessment.metrics()