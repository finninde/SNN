import numpy as np
from sklearn.linear_model import LinearRegression, Perceptron, LogisticRegression

class ReadoutLayer():
    def __init__(self, layer_type, dim):
        if layer_type == "linear":
            self.type = layer_type
            self.reg = LogisticRegression()

        elif layer_type == "perceptron":
            self.reg = Perceptron()
            self.type = layer_type

        else:
            self.type = "linear"
            self.reg = LogisticRegression()
        self.dim = dim

    def predict(self, X):
        if self.type == "linear":
            return self.reg.predict(X)
        elif self.type == "perceptron":
            return self.reg.predict(X)

    def train(self, X, y):
        if self.type == "linear":
            self.reg.fit(X,y)
        elif self.type == "perceptron":
            self.reg.fit(X,y)