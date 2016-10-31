"""Handy tools for visualization
"""

# Author: Jinchao Liu <liujinchao2000@gmail.com>
# License: BSD 

import numpy as np
from numpy.linalg import svd
from pylab import *
from matplotlib.patches import Ellipse
from matplotlib import cm
import matplotlib.pyplot as plt

__all__=[
    "create_covariance_ellipse"     
]


def create_covariance_ellipse(mean, cov):
    """Create an ellipse representing a covariance matrix
    Parameters:
    -----------
    mean: ndarray
    
    cov: ndarray
        
    """
    if not (isinstance(cov, (np.ndarray)) and \
            cov.shape[0] == cov.shape[1] and \
            cov.shape[0] == 2):
        raise ValueError("cov must be 2d array")
       
    U, S, V = svd(cov, full_matrices=True)
    theta = np.arccos(U[1,1])*(180.0/np.pi)    
    std = np.sqrt(S)
    e = Ellipse((mean[0], mean[1]), width=std[0]*3.0, height=std[1]*3.0, angle=theta)
    return e




if __name__ == "__main__":
    
    mean = [0.0, 0.0]
    cov = np.array([[5.0, 0.0],[0.0, 0.1]])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for i in range(50):        
        theta = i*5 * np.pi/180.0
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        rcov = np.asmatrix(R) * np.asmatrix(cov)

        re = create_covariance_ellipse(mean, rcov)
        ax.add_artist(re)
    
        re.set_alpha(0.5)
        color = np.array([0.3,0.1,0.6])*i/100.0
        re.set_facecolor(color)
    
    plt.axis('equal')
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

    