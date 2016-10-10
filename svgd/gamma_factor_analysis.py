import itertools

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import autograd.numpy as np
from autograd.numpy import trace
from autograd.numpy.linalg import det
from autograd.numpy.linalg import inv
from autograd.scipy.stats import norm
from autograd.scipy.special import gamma as gammaFn
from autograd.scipy.special import polygamma

from autograd import grad
from autograd import elementwise_grad as ew_grad


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
    d = x1 - x2
    squ_geo = np.sum( d*d )
    return (4*np.pi*params['t'])**(-n/2.) * np.exp((-.25/params['t']) * squ_geo)




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
    W_true = np.zeros((D,K)) + 0.001
    for d_idx in xrange(D):
        for k_idx in xrange(K):
            u = np.random.uniform()
            if u <= .2: W_true[d_idx, k_idx] = np.random.uniform(.001, .999)
    return W_true



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

# kernel
kernel = {'f': gamma_prob}
kernel['df'] = ew_grad(kernel['f'])

# initial distribution
q0 = np.random.gamma

### generate samples
n = 15
W_particles = []
for sampleIdx in xrange(n):
    W_particles.append(q0(shape=1, size=(D,K)))
    
    
maxEpochs = 5
lr = .001
for epochIdx in xrange(maxEpochs):
    for idx in xrange(n):
        W_particles[idx] += lr * steinOp(W_particles, idx, dLogPost, params, kernel)
        # check boundary condition, reflect if <= 0
        for k_idx in xrange(K):
            for d_idx in xrange(D):
                if W_particles[idx][d_idx,k_idx] <= 0: W_particles[idx][d_idx,k_idx] = np.abs(W_particles[idx][d_idx,k_idx])

print W_particles
