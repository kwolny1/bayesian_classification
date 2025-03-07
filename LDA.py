import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import norm

class LDA():
    """
    LDA implementation for binary data
    """
    def __init__(self):
        self.means = None
        self.sigma = None
        
        
    def fit(self, X, y): 
        self.means = [np.mean(X[y==0], axis=0), np.mean(X[y==1], axis=0)]

        S_0 = np.cov(X[y==0].T)
        S_1 = np.cov(X[y==1].T)
        self.sigma = (S_0 + S_1) / 2
        self.p = [1 - np.mean(y), np.mean(y)]

        self.w = np.linalg.inv(self.sigma).dot(self.means[1] - self.means[0])
        self.b = -0.5 * (self.means[0] + self.means[1]).dot(self.w) + np.log(len(X[y==0]) / len(X[y==1]))
        return self
        
    def predict_proba(self, X_test): 
        g = X_test.dot(self.w) + self.b
        prob_class_1 = 1 / (1 + np.exp(-g))
        prob_class_0 = 1 - prob_class_1
        return np.vstack([prob_class_0, prob_class_1]).T        
        
    def predict(self, X_test): 
        prob = self.predict_proba(X_test)
        return np.argmax(prob, axis=1)
        
    def get_params(self): 
        return self.means, self.sigma, self.p