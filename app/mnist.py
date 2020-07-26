from flask import render_template
import pickle

def mnist():
    return 'sdfdd'

from sklearn.datasets import fetch_openml
import numpy as np
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()
X, y = mnist["data"], mnist["target"]
X.shape
some_digit = X[0]
y = y.astype(np.uint8)
# X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# from sklearn.svm import SVC
# svm_clf = SVC()
# svm_clf.fit(X_train, y_train)
# svm_clf.predict([some_digit])

mnist_model = pickle.load(open("mnist_model.pkl", "rb"))
print(mnist_model)
