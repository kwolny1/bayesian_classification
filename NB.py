import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from scipy.stats import norm

class NB():
    """
    NB implementation for binary data
    """
    def __init__(self):
        self.means = None
        self.std = None
        self.p = None
        
    def fit(self, X, y):
        self.means = [np.mean(X[y==0], axis=0), np.mean(X[y==1], axis=0)]
        self.std = [np.std(X[y==0], axis=0), np.std(X[y==1], axis=0)]
        self.p = [1 - np.mean(y), np.mean(y)]
        
        return self
    
    def __norm(self, x, mean, cov): 
        # calculate normal distribution density for x
        return norm.pdf(x, mean, cov)
    
    def predict_proba(self, X_test): 
        if isinstance(X_test, pd.DataFrame): 
            X_test = X_test.to_numpy()
        res = []
        for x in X_test: 
            pr_1 = 1
            pr_0 = 1
            for feature in range(X_test.shape[1]): 
                pr_1 = pr_1 * self.__norm(x[feature], self.means[1].iloc[feature], self.std[1].iloc[feature])
                pr_0 = pr_0 * self.__norm(x[feature], self.means[0].iloc[feature], self.std[0].iloc[feature])
            gora = pr_1*self.p[1]
            dol = pr_1*self.p[1] + pr_0*(1 - self.p[1])
            
            prob_cond = gora/dol
            res.append([1-prob_cond, prob_cond])
        return res
        
    def predict(self, X_test):
        prob = self.predict_proba(X_test)
        return np.argmax(prob, axis=1)
        
    def get_params(self): 
        return self.means, self.std, self.p