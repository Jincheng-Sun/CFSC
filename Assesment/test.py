import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import label_binarize
# from sklearn.model_selection import train_test_split
#
# # 3 classes to classify
# n_classes = 3
#
# X, y = make_classification(n_samples=80000, n_features=20, n_informative=3, n_redundant=0, n_classes=n_classes,
#     n_clusters_per_class=2)
#
# # Binarize the output
# y = label_binarize(y, classes=[0, 1, 2])
# n_classes = y.shape[1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
#
# from keras.models import Sequential
# from keras.layers import Dense
#
# def build_model():
#     model = Sequential()
#     model.add(Dense(20, input_dim=20, activation='relu'))
#     model.add(Dense(40, activation='relu'))
#     model.add(Dense(3, activation='softmax'))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# keras_model2 = build_model()
# keras_model2.fit(X_train, y_train, epochs=10, batch_size=100, verbose=1)
#
# y_score = keras_model2.predict(X_test)


a = np.arange(5)
print(a)
b = map(lambda x:'class %d'%x,a.tolist())