"""Preprocessing
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 3 clause


import numpy as np
import numbers
import copy


def normalize(X, norm='l2'):
    """Input samples are transformed to have unit L2 norm

    Parameters:
    -----------
    X : array-like [n_samples, n_features]
    
    Return:
    -------
    tran_X : array-like [n_samples, n_features] which has unit L2 norm 
    """
    X_t = copy.copy(X)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    X_t = X_t.astype(np.float64) # For better accuracy
    
    norm = np.sqrt(np.sum(X_t * X_t, axis=0))
    
    X_t = np.delete(X_t, np.where(norm==0), axis=1)
    norm = np.delete(norm, np.where(norm==0))   
    
    X_t /= np.reshape(norm, (1, len(norm))) 
    return X_t
    

def standardize(X):
    """Input samples are transformed so that each feature has zero mean and unit variance. 
    
    Parameters:
    -----------
    X : array-like [n_samples, n_features]
    
    Return:
    -------
    tran_X : array-like [n_samples, n_features] which has zero mean and unit variance
    """
    # Check parameters 
    X_t = copy.copy(X)
    n_samples = X.shape[0]
    n_features = X.shape[1]
    X_t = X_t.astype(np.float64)
        
    # Subtract mean
    X_t -= np.reshape(np.mean(X_t, axis=0),(1,n_features))    
    
    # Unit variance
    std = np.std(X_t, axis=0)        
    
    # Delete zero-std rows
    X_t = np.delete(X_t, np.where(std==0), axis=1)
    std = np.delete(std, np.where(std==0))        
    
    X_t /= np.reshape(std,(1, len(std)))    
    return X_t
    
