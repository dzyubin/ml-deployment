import psycopg2
import pickle
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC

# conn = psycopg2.connect(host="ec2-54-75-244-161.eu-west-1.compute.amazonaws.com",database="d790i0bj2ikkeq",user="qumbrzpxinjjas",password="157959a5cf68334fcb38a2e3df6f65b97e82186eaa9c21ce6e80c6e1a4aff253")
# cur = conn.cursor()

# cur.execute('select model from models')
# bytes_model = cur.fetchone()
# model = pickle.loads(bytes_model[0])
# print(model.get_params())

mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]

some_digit = X[0]

y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# svm_clf = SVC()
# svm_clf.fit(X_train, y_train)
# svm_clf.predict([some_digit])

