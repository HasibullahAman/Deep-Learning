import numpy as np


class NearistNighbors:
    def __init__(self):
        pass

    def train(self, X,y):
        self.Xtr = X
        self.ytr = y

    def predict(self,X):
        num_test = X.shape[0]
        ypred = np.zeros(num_test,dtype=self.ytr.dtype)


        for i in xrange(num_test):
            distence = np.sum(np.abs(self.Xtr - X[i,:]),axis=1)
            min_index = np.argmin(distence)
            ypred[i] = self.ytr[min_index]
            return ypred