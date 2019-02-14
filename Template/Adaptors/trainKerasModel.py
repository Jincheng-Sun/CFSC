from Template.Adaptors.trainAdaptor import TrainAdaptor
class TrainKerasModel(TrainAdaptor):
    def __init__(self, model, X_train, Y_train, X_val, Y_val):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.model = model

    def train(self, batch_size, epoch,call_backs=None,loss='categorical_crossentropy', optimizer='adam'):

        self.model.compile(loss=loss,
                           optimizer=optimizer,
                           metrics=['accuracy'])

        self.model.fit(self.X_train, self.Y_train,
                       batch_size=batch_size,
                       epochs=epoch,)
                       # validation_data=(self.X_val, self.Y_val),
                       # callbacks=[call_backs])

    def save(self, model_path):
        self.model.save(model_path)
