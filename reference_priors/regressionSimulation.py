import numpy as np
from matplotlib import pyplot as mp
import random
import sklearn
from sklearn.datasets import load_iris
import pylab as P
import math
import matplotlib.mlab as mlab


def load_iris_data():
    # load iris data
    # take third dimension since it's interestingly bi-modal
    iris = load_iris()
    n_targets = len(np.unique(iris.target))
    data = np.array(iris.data[:,2:4])
    return data, n_targets


def jeffPrior(X, sigma):
    return det(np.dot(X,X.T))**(d/2)/sigma


if __name__=="__main__":
    
    X, n_targets = load_iris_data()
    # values for Gaussians                                                                                                                                    
    vals = np.linspace(363,763, 100000)
        
    

    P.figure()
    # plot the histogram of these means
    P.scatter(np.array(explicit_bagged_sums[:,0].T)[0], np.array(explicit_bagged_sums[:,1].T)[0], s=30, c='k', alpha=0.45)

    # plot a Gaussian fit to the sums
    emp_mean = explicit_bagged_sums.mean(axis=0)
    emp_variance = np.zeros((2,2))
    for idx in xrange(explicit_bagged_sums.shape[0]):
        emp_variance += np.dot((explicit_bagged_sums[idx,:] - emp_mean).T, (explicit_bagged_sums[idx,:] - emp_mean))
    emp_variance *= 1./no_MC_iterations

    delta = .25
    x = np.arange(450, 670, delta)
    y = np.arange(130, 230, delta)
    X, Y = np.meshgrid(x, y)
    Z = mlab.bivariate_normal(X, Y, np.sqrt(emp_variance[0,0]),  np.sqrt(emp_variance[1,1]), emp_mean[0,0], emp_mean[0,1], sigmaxy=emp_variance[0,1])
    CS = P.contour(X, Y, Z, linewidths=3)
    P.title("Empirical Distribution")

    P.show()


