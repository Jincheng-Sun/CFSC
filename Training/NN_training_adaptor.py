from Training.Training_adaptor import Training_adaptor
import numpy as np


class NN_training_adaptor(Training_adaptor):

    def __init__(self):
        super().__init__()

    def load_data(self, x_train, y_train, x_test, y_test, **kwargs):

        self.X_train = np.load(x_train)
        self.Y_train = np.load(y_train)
        self.X_test = np.load(x_test)
        self.Y_test = np.load(y_test)
        self.Data_dim = self.X_train.shape[1]

    def create_dataset(self, **kwargs):
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        from keras.utils.np_utils import to_categorical
        Y_train = []
        Y_test = []
        expand_class = None
        try:
            expand_class = kwargs['expand_class']
        except:
            print('[INFO]: No label expanded')

        if expand_class != None:
            for fl, sl in zip(self.Y_train[:, 0], self.Y_train[:, 1]):
                if fl == expand_class:
                    Y_train.append(sl+5)
                else:
                    Y_train.append(fl)
            Y_train = np.array(Y_train)

            for fl, sl in zip(self.Y_test[:, 0], self.Y_test[:, 1]):
                if fl == expand_class:
                    Y_test.append(sl)
                else:
                    Y_test.append(fl)
            Y_test = np.array(Y_test)
        else:
            Y_train = self.Y_train[:, 0]
        Y_train = pd.DataFrame(Y_train)[0]
        x_train, x_val, y_train, y_val = train_test_split(self.X_train, Y_train, test_size=0.2, random_state=42)
        # y_train = pd.DataFrame(y_train)[0]
        # y_val = pd.DataFrame(y_val)[0]
        # one-hot，5 category
        y_labels = list(Y_train.value_counts().index)
        # y_labels = np.unique(y_train)
        le = preprocessing.LabelEncoder()
        le.fit(y_labels)
        num_labels = len(y_labels)
        y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
        y_val = to_categorical(y_val.map(lambda x: le.transform([x])[0]), num_labels)

        return num_labels, x_train, y_train, x_val, y_val, Y_test

    def network(self, **kwargs):

        from keras.models import Sequential
        from keras.layers import Dense, Dropout
        num = kwargs['cls_num']
        model = Sequential()
        model.add(Dense(1024, input_shape=(self.Data_dim,), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num, activation='softmax'))
        model.summary()
        return model

    def train_model(self, model, X_train, Y_train, **kwargs):
        from keras.callbacks import EarlyStopping

        x_val = kwargs['x_val']
        y_val = kwargs['y_val']
        model_path = kwargs['save_path']
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        # model = models.load_model('0.7305NN')
        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
        model.fit(X_train, Y_train,
                  batch_size=500,
                  epochs=50,
                  validation_data=(x_val, y_val),
                  callbacks=[monitor])
        model.save(model_path)
        return model

    def load_model(self, model):
        from keras.models import load_model
        model = load_model(model)
        return model

    def assesment(self, model, X_test, Y_test):
        from sklearn.metrics import  accuracy_score
        y_pred = model.predict_classes(X_test)
        score = accuracy_score(y_pred, Y_test)
        print(score)
