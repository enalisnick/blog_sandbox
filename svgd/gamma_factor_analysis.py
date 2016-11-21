import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import autograd.numpy as np
from autograd.numpy import trace
from autograd.numpy.linalg import det
from autograd.numpy.linalg import inv
from autograd.numpy.linalg import norm as vecNorm
from autograd.scipy.stats import norm
from autograd.scipy.special import gamma as gammaFn
from autograd.scipy.special import polygamma

from autograd import grad
from autograd import elementwise_grad as ew_grad

from kernels import *


### Stein Operator
def steinOp(x, idx, dLogModel, params, kernel, kernelParams=None):
    returnVal = 0.
    n = len(x)
    for j in xrange(n):
        returnVal += kernel['f'](x[j], x[idx]) * dLogModel(x[j],params) + kernel['df'](x[j], x[idx])
    return 1./n * returnVal


### Kernels

# Diffusion Kernel for Gamma r.v.'s via Parametrix Expansion
# Dropped \Psi terms following (Lafferty & Lebanon, 2005)
'''
def diffusion(x1, x2, params={'t': .01}):
    try: n = len(x1) 
    except: n = 1
    geo = polygamma(1,x1)-polygamma(1,x2)
    squ_geo = np.sum( geo*geo )
    return (4*np.pi*params['t'])**(-n/2.) * np.exp((-.25/params['t']) * squ_geo)


def gamma_prob(x1, x2, params={'p':.5}):
    return gammaFn(params['p']*(x1+x2-2.) + 1.) / (gammaFn(x1)*gammaFn(x2))**params['p']

def rbf(x1, x2, params={'t': 1.}):
    try: n = len(x1)
    except: n = 1
    x1 = np.log(x1)
    x2 = np.log(x2)
    d = x1 - x2
    squ_geo = np.sum( d*d )
    return (4*np.pi*params['t'])**(-n/2.) * np.exp((-.25/params['t']) * squ_geo)
'''



# Factor Analysis Model
def logPostFA(W, params):
    X = params['data']
    N = X.shape[0]
    D,K = W.shape
    Q = np.dot(W, W.T) + np.eye(D)*.1
    log_prior = D*K*params['a']*np.log(params['b']) - D*K*np.log(gammaFn(params['a'])) + np.sum((params['a']-1)*np.log(W)) - np.sum(params['b']*W)
    return N/2. * np.log(det(Q)) - .5*trace(np.dot(np.dot(X.T,X), inv(Q))) + log_prior

# Simulate data and true load matrix
def simulateData(W, N=1000, K=50):
    Z = np.random.normal(size=(N,K))
    return np.dot(Z,W.T)

def construct_true_W(D, K):
    W_true = np.zeros((D,K)) + 0.01
    for d_idx in xrange(D):
        for k_idx in xrange(K):
            u = np.random.uniform()
            if u <= .2: W_true[d_idx, k_idx] = np.random.uniform(.001, .999)
    return W_true


# Eval utilities

def amariError(W_hat, W_true):
    A = np.dot(W_hat.T, W_true)
    n_rows, n_cols = A.shape
    
    amari_error = 0.
    for col_idx in xrange(n_cols):
        row_max = np.amax(np.abs(A[:, col_idx]))
        amari_error += (1./(2*n_cols)) * (np.sum(np.abs(A[:, col_idx])) / row_max - 1.)
    
    for row_idx in xrange(n_rows):
        col_max = np.amax(np.abs(A[row_idx,:]))
        amari_error += (1./(2*n_rows)) * (np.sum(np.abs(A[row_idx, :])) / col_max - 1.)

    return amari_error



if __name__ == '__main__':

    K = 50
    D = 100
    W_true = construct_true_W(D, K)

    logPost = logPostFA
    dLogPost = grad(logPost)
    params = {
        'data': simulateData(W_true),
        'a': .5,
        'b': 1.
       }

    # initial distribution                                                                                                                                                        
    q0 = np.random.gamma

    maxEpochs = 25
    lr = .001
    n_particles = [3, 5, 10, 15, 20]#, 25, 30, 40, 50]
    kernels = [gamma_prob, log_rbf, gamma_kld, gamma_diffusion] #, composite, diffusion]
    
    results = []
    for k in kernels:
        results.append([])

        # kernel                                                                                                                                                                     
        kernel = {'f': k}
        kernel['df'] = ew_grad(kernel['f'])
        
        for n in n_particles:
    
            ### generate samples
            W_particles = []
            for sampleIdx in xrange(n):
                W_particles.append(q0(shape=1, size=(D,K)))
        
            for epochIdx in xrange(maxEpochs):
                for idx in xrange(n):
                    W_particles[idx] += lr * steinOp(W_particles, idx, dLogPost, params, kernel)
                    # check boundary condition, reflect if <= 0
                    for k_idx in xrange(K):
                        for d_idx in xrange(D):
                            if W_particles[idx][d_idx,k_idx] <= 0: W_particles[idx][d_idx,k_idx] = np.abs(W_particles[idx][d_idx,k_idx])
                            if W_particles[idx][d_idx,k_idx] < 0.0001: W_particles[idx][d_idx,k_idx] = 0.0001

            W_avg = np.zeros(W_particles[0].shape)
            c = 0
            for w in W_particles: 
                if not (np.any(np.isnan(w)) or np.any(np.isinf(w))):
                    W_avg += w
                    c += 1      
            W_avg /= c

            results[-1].append(amariError(W_avg, W_true))


    plt.figure()
    plt.plot(n_particles, results[0], 'kx-', markersize=9, mew=3, linewidth=3, label="Probability Product Kernel")
    plt.plot(n_particles, results[1], 'bx-', markersize=9, mew=3, linewidth=3, label="Log RBF Kernel")
    plt.plot(n_particles, results[2], 'cx-', markersize=9, mew=3, linewidth=3, label="KLD Kernel")
    plt.plot(n_particles, results[3], 'rx-', markersize=9, mew=3, linewidth=3, label="Info Diffusion Kernel")
    plt.xlabel("Number of SVGD Particles")
    plt.ylabel("Amari Error")
    plt.legend()
    plt.title("Gamma Factor Analysis Simulation: Amari Error vs Number of SVGD Particles")
    plt.show()
