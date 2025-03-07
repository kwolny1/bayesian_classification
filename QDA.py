import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import norm

class QDA():
    """
    QDA implementation for binary data
    """
    def __init__(self):
        self.means = None
        self.sigma = None
        self.p = None
        
    def fit(self, X, y):
        self.means = [np.mean(X[y==0], axis=0), np.mean(X[y==1], axis=0)]
        self.sigma = [np.cov(X[y==0].T), np.cov(X[y==1].T)]
        self.p = [1 - np.mean(y), np.mean(y)]
        
        return self
    
    def __mvn(self, x, mean, cov): 
        # calculate multivariate normal distribution
        return multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(x)
        
    def predict_proba(self, X_test): 
        if isinstance(X_test, pd.DataFrame): 
            X_test = X_test.to_numpy()
        res = []
        for x in X_test: 
            nominator = self.__mvn(x, self.means[1], self.sigma[1])*self.p[1]
            denominator = self.__mvn(x, self.means[1], self.sigma[1])*self.p[1] + self.__mvn(x, self.means[0], self.sigma[0])*(1 - self.p[0])
            prob_cond = nominator/denominator
            res.append([1-prob_cond, prob_cond])
        return res
        
    def predict(self, X_test):
        prob = self.predict_proba(X_test)
        return np.argmax(prob, axis=1)
        
    def get_params(self): 
        return self.means, self.sigma, self.p