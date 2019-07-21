import numpy as np
from sklearn.linear_model import LinearRegression, Perceptron

class ReadoutLayer():
    def __init__(self, layer_type, dim):
        if layer_type == "linear":
            self.type = "linear"
            self.reg = LinearRegression()

        elif layer_type == "perceptron":
            self.reg = Perceptron()
            self.type = layer_type

        else :
            self.type = "linear"
            self.reg = LinearRegression()
        self.dim = dim

    def predict(self, X):
        return self.reg.predict(X)

    def train(self, X, y):
        self.reg.fit(X,y)