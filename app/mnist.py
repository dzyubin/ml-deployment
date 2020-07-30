from flask import render_template
import pickle

import tensorflow as tf
from tensorflow import keras
# import matplotlib.pyplot as plt

# from sklearn.datasets import fetch_openml
# import numpy as np
# mnist = fetch_openml('mnist_784', version=1)
# mnist.keys()
# X, y = mnist["data"], mnist["target"]
# X.shape
# some_digit = X[0]
# y = y.astype(np.uint8)
# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# from sklearn.svm import SVC
# svm_clf = SVC()
# svm_clf.fit(X_train, y_train)
# svm_clf.predict([some_digit])

# mnist_model = pickle.load(open("mnist_model.pkl", "rb"))
# print(mnist_model)

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
print('\n\n\n')
print(X_train_full.shape)
print('\n\n\n')

X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.

# model = keras.models.load_model("models/my_mnist_model.h5") # rollback to best model

#X_new = X_test[:3]
#y_proba = model.predict(X_new)
#print(y_proba.round(2))
#print(y_test[:3])

#y_pred = model.predict_classes(X_new)
#y_pred

print('\n\n')
#print(X_train_full.shape)
# print(model.get_config())
print('\n\n')

def mnist():
	print('sdf')
