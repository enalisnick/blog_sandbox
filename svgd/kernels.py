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

### GENERAL KERNELS ###

def kld(x1, x2, kldFn, params={'alpha':3, 'beta':0}):
    sym_kld = kldFn(x1, x2) + kldFn(x2, x1)
    return np.exp( -params['alpha']*sym_kld + params['beta'] )

def rbf(x1, x2, params={'t': .09}):
    try: n = len(x1)
    except: n = 1
    d = x1 - x2
    squ_geo = np.sum( d*d )
    return np.exp((-.25/params['t']) * squ_geo)


### GAMMA KERNELS ###

# Diffusion Kernel for Gamma r.v.'s via Parametrix Expansion                                                                           
# Dropped \Psi terms following (Lafferty & Lebanon, 2005)                                                    
def gamma_diffusion(x1, x2, params={'t': .09}):
    try: n = len(x1)
    except: n = 1
    geo = polygamma(1,x1)-polygamma(1,x2)
    squ_geo = np.sum( geo*geo )
    return np.exp((-.25/params['t']) * squ_geo)

def gamma_score(x, params):
    v1 = np.log(x) - np.log(params['theta']) - polygamma(0, params['alpha'])
    v2 = x/params['theta']**2 - params['alpha']/params['theta']
    return np.array([[v1, v2]])

def gamma_fisher(x1, x2, params={'alpha': 1., 'theta': 1.}):
    score1 = gamma_score(x1, params)
    score2 = gamma_score(x2, params)
    temp = 1./params['theta']
    fisher_info_mat = np.array([[polygamma(1,params['alpha']), temp],[temp, params['alpha']/params['theta']**2]])
    return np.dot(np.dot(score1, inv(fisher_info_mat)), score2.T)[0,0]

def gamma_prob(x1, x2, params={'p':.5}):
    return gammaFn(params['p']*(x1+x2-2.) + 1.) / (gammaFn(x1)*gammaFn(x2))**params['p']

def composite_rbf(x1, x2, params={'t':.09, 'b':1.5}):
    #if x1 > 1: f1 = np.exp(1) - 1
    #else: f1 = np.exp(x1/params['b']) - 1
    #if x2 > 1: f2 = np.exp(1) - 1
    #else: f2 = np.exp(x2/params['b']) - 1
    return .08*x1*rbf(x1,x2,params)*x2

def gamma_kld(a1, a2, theta1=10., theta2=10.):
    return np.sum((a1-a2)*polygamma(0,a1) - np.log(gammaFn(a1)) + np.log(gammaFn(a2)) + a2*(np.log(theta2)-np.log(theta1)) + a1*((theta1-theta2)/theta2))

def log_rbf(x1, x2, params={'t': .09}):
    return rbf(np.log(x1), np.log(x2), params)


### BETA KERNELS ###
def beta_score(x, params):
    v1 = np.log(x) - polygamma(0,params['alpha']) + polygamma(0,params['alpha']+params['beta'])
    v2 = np.log(1-x) - polygamma(0,params['beta']) + polygamma(0,params['alpha']+params['beta'])
    return np.array([[v1, v2]])

def beta_fisher(x1, x2, params={'alpha':.5,'beta':.5}):
    score1 = beta_score(x1, params)
    score2 = beta_score(x2, params)
    temp = polygamma(1,params['alpha']+params['beta'])
    fisher_info_mat = np.array([[polygamma(1,params['alpha']) - temp, -temp],[-temp, polygamma(1,params['beta']) - temp]])
    return np.dot(np.dot(score1, inv(fisher_info_mat)), score2.T)[0,0]

def bernoulli_prob_prod(x1, x2, params={'p':.75}):
    return (x1*x2)**params['p'] + (1-x1)**params['p'] * (1-x2)**params['p']

def unit_sobolev(x1, x2):
    return 2*(np.minimum(x1,x2) - x1*x2)

def bernoulli_kld(p1, p2):
    return p1 * np.log(p1/p2) + (1-p1) * np.log((1-p1)/(1-p2))
