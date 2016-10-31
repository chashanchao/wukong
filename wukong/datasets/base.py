"""
    @brief      Load all the machine learning dataset
    @author     Jinchao Liu
    @date       Jan 19, 2015
    @version    0.0.0.1
"""

import numpy as np
import copy
import matplotlib.pyplot as plt
import unittest
import os


__all__ = [
    "load_dataset", 
    "load_toy_example" 
] 


########################################################################
# Public interfaces
def load_dataset(ds_name, noise_level = 0.0):
    if ds_name.lower() == 'abalone':
        X,y = load_abalone()
    elif ds_name.lower() == 'abalone3c':
        X,y = load_abalone3c()
    elif ds_name.lower() == 'abalone2c':
        X,y = load_abalone2c()
    elif ds_name.lower() == 'letter':
        X,y = load_letter()
    elif ds_name.lower() == 'ionosphere':
        X,y = load_ionosphere()
    elif ds_name.lower() == 'german':
        X,y = load_german()   
    elif ds_name.lower() == 'australian':
        X,y = load_australian()
    elif ds_name.lower() == 'heart':
        X,y = load_heart()
    elif ds_name.lower() == 'pima':
        X,y = load_pima()
    elif ds_name.lower() == 'banana':
        X,y = load_banana()
    else:
        print 'This dataset does not exist'

    y = flip_label_randomly(y, noise_level)
    return X, y 
    

def load_toy_example(ds_name, 
                     noise_level = 0.0,
                     n_pos_samples = 100,
                     n_neg_samples = 100):
    if ds_name.lower() == 'toy0':
        return load_toy0(noise_level, n_pos_samples, n_neg_samples)
    
    if ds_name.lower() == 'toy1':
        return load_toy1(noise_level, n_pos_samples, n_neg_samples)

    if ds_name.lower() == 'toy2':
        return load_toy2(noise_level, n_pos_samples, n_neg_samples)
    
    else:
        print 'This dataset does not exist'    
    
    
def random_flip_label(X_positive, X_negative, noise_level):
    X_p_withnoise = []
    X_n_withnoise = []
    
    n_samples = X_positive.shape[0]
    for i in range(n_samples):
        if np.random.rand() < noise_level:
            X_n_withnoise.append(X_positive[i,:])
        else:
            X_p_withnoise.append(X_positive[i,:])

    n_samples = X_negative.shape[0]
    for i in range(n_samples):
        if np.random.rand() < noise_level:
            X_p_withnoise.append(X_negative[i,:])
        else:
            X_n_withnoise.append(X_negative[i,:])

    return np.array(X_p_withnoise), np.array(X_n_withnoise)



def flip_label_randomly(y, noise_level):
    y_withnoise = copy.copy(y)
    n_samples = len(y)
    for i in range(n_samples):
        if np.random.rand() < noise_level:
            y_withnoise[i] *= -1 # Assume that y \in {-1,1}
    
    return y_withnoise




def split_pos_neg(X, y):
    """Must be two-classes"""
    X_pos = X[np.where(y==1), :]
    X_neg = X[np.where(y==-1), :]

    return X_pos, X_neg
    
    

########################################################################
# Implementation
if os.name == 'nt':
    direc = r'C:/Users/ljc/Dropbox/ensemble/mlnp/datasets/'
elif os.name == 'posix':
    direc = '/home/jcbigtree/Dropbox/ensemble/mlnp/datasets/'

def load_abalone():
    ds_path = direc + 'abalone/abalone.data'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split(',')
        if words[0] == 'M': words[0] = 0
        if words[0] == 'F': words[0] = 1
        if words[0] == 'I': words[0] = 2
        X.append([float(x) for x in words[0:8]])
        y.append(words[8:9])
    f.close()
    y = [float(yi[0]) for yi in y]
    return X, y


def load_abalone3c():
    X, y = load_abalone()
    for k in range(len(y)):
        if   y[k] < 9:  y[k] = 0
        elif y[k] < 18: y[k] = 1
        else:           y[k] = 2 
    return X, y


def load_abalone2c():
    X, Y = load_abalone()
    for k in range(len(Y)):
        if   Y[k] < 10: Y[k] = -1
        else:           Y[k] = 1 
    return X, Y


def load_banana():
    ds_path = direc + 'banana/banana.data'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split(',')
        X.append([float(x) for x in words[0:2]])
        y.append(words[2])
    f.close()
    y = np.array([float(yi) for yi in y])
    X = np.array(X)
    return X, y


def load_letter():
    ds_path = direc + 'letter/letter-recognition.data'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split(',')
        words[0] = ord(words[0])%32
        X.append([float(x) for x in words[1:]])
        y.append(words[0])
    f.close()
    y = [float(yi) for yi in y]
    return X, y


def load_ionosphere():
    ds_path = direc + 'ionosphere/ionosphere.data'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split(',')
        X.append([float(x) for x in words[0:-1]])
        if words[-1] == 'b': y.append(-1.0)
        if words[-1] == 'g': y.append(1.0)
    f.close()
    return X, y


def load_german():
    ds_path = direc + 'german/german.data-numeric'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split()
        X.append([float(x) for x in words[0:-1]])
        if words[-1] == '2': y.append(-1.0)
        if words[-1] == '1': y.append(1.0)
    f.close()
    return X, y
    

def load_australian():
    ds_path = direc + 'australian/australian.data'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split()
        X.append([float(x) for x in words[0:-1]])
        if words[-1] == '0': y.append(-1.0)
        if words[-1] == '1': y.append(1.0)
    f.close()
    return X, y


def load_heart():
    ds_path = direc + 'heart/processed.cleveland.data'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split(',')
        X.append([float(x) for x in words[0:-1]])
        if words[-1] == '0': y.append(-1.0)
        else:                y.append(1.0)
    f.close()
    return X, y


def load_pima():
    ds_path = direc + 'pima/pima_indians_diabetes.data'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split(',')
        X.append([float(x) for x in words[0:-1]])
        if words[-1] == '0': y.append(-1.0)
        else:                y.append(1.0)
    f.close()
    return X, y
    

def load_iris():
    ds_path = direc + 'iris/iris.data'
    f = open(ds_path,'r')
    X = []  # input variables
    y = []  # target variables
    for line in f:
        words = line.strip().split(',')
        X.append([float(x) for x in words[0:-1]])
        if words[-1] == '0': y.append(-1.0)
        else:                y.append(1.0)
    f.close()
    return X, y
        

def load_toy0(noise_level=0.0, n_pos_samples=100, n_neg_samples=100):
    """create a toy 2D example which can be illustrated intutively"""
    #Need to check parameters first
    s = 0.01    
    mu0 = 0.25,0.25
    cov0 = [[s,0],[0,s]]
    X0 = np.random.multivariate_normal(mu0, cov0, n_pos_samples/2)
    y0 = [1.0]*n_pos_samples

    mu1 = 0.75,0.25
    cov1 = [[s,0],[0,s]]
    X1 = np.random.multivariate_normal(mu1, cov1, n_neg_samples/2)
    y1 = [-1.0]*n_neg_samples
    
    mu2 = 0.75,0.75
    cov2 = [[s,0],[0,s]]
    X2 = np.random.multivariate_normal(mu2, cov2, n_pos_samples/2)
    y2 = [1.0]*n_pos_samples
    
    mu3 = 0.25,0.75
    cov3 = [[s,0],[0,s]]
    X3 = np.random.multivariate_normal(mu3, cov3, n_neg_samples/2)
    y3 = [-1.0]*n_neg_samples
                      
    X_positive = np.concatenate((X0,X2),axis=0)
    X_negative = np.concatenate((X1,X3),axis=0)
    
    return  random_flip_label(X_positive, X_negative, noise_level)


def load_toy1(noise_level=0.0, n_pos_samples=100, n_neg_samples=100):
    """create a toy 2D example which can be illustrated intutively"""
    s = 0.01    
    mu0 = 0.25,0.25
    cov0 = [[s,0],[0,s]]
    X0 = np.random.multivariate_normal(mu0, cov0, n_pos_samples)
    y0 = [1.0]*n_pos_samples

    mu1 = 0.75,0.25
    cov1 = [[s,0],[0,s]]
    X1 = np.random.multivariate_normal(mu1, cov1, n_neg_samples)
    y1 = [-1.0]*n_neg_samples
    
    X_positive = X0
    X_negative = X1
    
    return  random_flip_label(X_positive, X_negative, noise_level)



def load_toy2(noise_level=0.0, n_pos_samples=100, n_neg_samples=100):
    s = 0.01    
    mu0 = 0.25,0.25
    cov0 = [[0.01, 0],[0, 0.5]]
    X0 = np.random.multivariate_normal(mu0, cov0, n_pos_samples)
    y0 = [1.0]*n_pos_samples

    mu1 = 0.75,0.25
    cov1 = [[0.01, 0],[0, 0.5]]
    X1 = np.random.multivariate_normal(mu1, cov1, n_neg_samples)
    y1 = [-1.0]*n_neg_samples
   
    theta = np.pi * 0.25
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.cos(theta), np.sin(theta)]])
    X_positive = np.dot(X0, R)
    X_negative = np.dot(X1, R)
    
    return  random_flip_label(X_positive, X_negative, noise_level)


###################################################################################################
# Unit test
###################################################################################################
class TestDatasetLoader(unittest.TestCase):

    def test_load_toy(self):        
        X_p, X_n = load_toy_example("toy0")        
        #colors = ['blue','red']
        #for i in range(X.shape[0]):
        #    color_index = int((y[i]+1)*0.5)
        #    plt.plot(X[i,0],X[i,1], 'o', color=colors[color_index])
        
        X = np.concatenate((X_p,X_n))
        
        plt.plot(X_p[:,0],X_p[:,1], 'o', color='green')
        plt.plot(X_n[:,0],X_n[:,1], 'o', color='red')
        plt.title("2D Toy example")
        xymin = np.min(X,axis=0) - 0.1
        xymax = np.max(X,axis=0) + 0.1
        plt.xlim((xymin[0], xymax[0]))
        plt.ylim((xymin[1], xymax[1]))
        plt.show()
   
    
def main():
    unittest.main()    


if __name__ == "__main__":
    main()
   
    
    
    
    
    
    
    
    
    
    
    
    
        
        
