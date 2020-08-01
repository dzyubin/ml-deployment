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

# (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
# print('\n\n\n')
# print(X_train_full.shape)
# print('\n\n\n')

# X_valid, X_train = X_train_full[:5000] / 255., X_train_full[5000:] / 255.
# y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# X_test = X_test / 255.

model = keras.models.load_model("models/my_mnist_model.h5") # rollback to best model

# '5'
some_digit = [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   3.,  18.,
        18.,  18., 126., 136., 175.,  26., 166., 255., 247., 127.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
        30.,  36.,  94., 154., 170., 253., 253., 253., 253., 253., 225.,
       172., 253., 242., 195.,  64.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,  49., 238., 253., 253., 253., 253.,
       253., 253., 253., 253., 251.,  93.,  82.,  82.,  56.,  39.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
        18., 219., 253., 253., 253., 253., 253., 198., 182., 247., 241.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,  80., 156., 107., 253.,
       253., 205.,  11.,   0.,  43., 154.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,  14.,   1., 154., 253.,  90.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
       139., 253., 190.,   2.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,  11., 190., 253.,  70.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,  35., 241., 225., 160., 108.,   1.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  81., 240.,
       253., 253., 119.,  25.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,  45., 186., 253., 253., 150.,  27.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,  16.,  93., 252., 253., 187.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 249., 253.,
       249.,  64.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,  46., 130., 183., 253., 253., 207.,   2.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,  39., 148., 229., 253., 253., 253.,
       250., 182.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  24., 114.,
       221., 253., 253., 253., 253., 201.,  78.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,  23.,  66., 213., 253., 253., 253., 253., 198.,  81.,
         2.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,  18., 171., 219., 253., 253.,
       253., 253., 195.,  80.,   9.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  55.,
       172., 226., 253., 253., 253., 253., 244., 133.,  11.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0., 136., 253., 253., 253., 212., 135.,
       132.,  16.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
         0.,   0.,   0.]

y_proba = model.predict(some_digit)
print(y_proba.round(2))

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
