import numpy as np
from typing import Union
from sklearn.base import BaseEstimator, ClassifierMixin


class GaussianBayes(BaseEstimator, ClassifierMixin):
    """ Classification by normal law by Bayesian approach
    """
    def __init__(self, priors:np.ndarray=np.array([]), diagonal:bool=False) -> None:
        self.priors = priors    # a priori probabilities of classes
                                # (n_classes,)

        self.mu = None          #  mean of each feature per class
                                # (n_classes, n_features)
        self.sigma =  None      # covariance of each feature per class
                                # (n_classes, n_features, n_features)
                                
        self.diagonal = diagonal
    
    
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        """
        X shape = [n_samples, n_features]
        maximum log-likelihood
        """
        n_obs = X.shape[0]
        n_classes = len(self.mu)
        n_features = X.shape[1]

        # initalize the output vector
        y = np.zeros((n_obs,), dtype=int)
        det = np.array([np.linalg.det(self.sigma[i]) for i in range(n_classes)])
        inv = np.array([np.linalg.inv(self.sigma[i]) for i in range(n_classes)])
        
        # for each point
        for j in range(n_obs):
            best = float('inf')
            
            # find most likely class
            for i in range(n_classes):
                diff = X[j] - self.mu[i]
                tmp = np.log(det[i]) + np.dot(np.dot(diff.T, inv[i]), diff)
                
                # add a priori probability
                if self.priors.size != 0:
                    tmp += np.log(self.priors[i])
                
                # update prediction
                if tmp < best:
                    y[j] = i
                    best = tmp
        
        return y 
    
    
    
    def fit(self, X:np.ndarray, y:np.ndarray) -> None:
        """Learning of parameters
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        # number of random variables and classes
        n_classes = len(np.unique(y))
        n_features = X.shape[1]
        
        # split data
        X_split = [np.array([data for (data, label) in zip(X, y) if label == i]) for i in range(n_classes)]
        
        # learning
        self.mu = [np.mean(X_split[i], axis=0) for i in range(n_classes)]
        #print(self.mu)
        self.sigma = np.array([np.cov(X_split[i], rowvar=False) * np.identity(n_features) if self.diagonal else np.cov(X_split[i], rowvar=False) for i in range(n_classes)])
        

    def score(self, X:np.ndarray, y:np.ndarray) -> float:
        """Compute the precision
        X : shape (n_data, n_features)
        y : shape (n_data)
        """
        return np.sum(y == self.predict(X)) / len(X)
