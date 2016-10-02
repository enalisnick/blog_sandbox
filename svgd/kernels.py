import numpy as np
from scipy.linalg import inv
from scipy.special import gamma
from scipy.special import polygamma

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

def gamma_fisher(x1, x2, params={'alpha': .1}):
    term1 = polygamma(0,params['alpha']) - np.log(params['alpha'])
    term1 *= term1/polygamma(1,params['alpha'])
    term2 = (x1-1)*(x2-1)/params['alpha']
    return term1 + term2

def gamma_prob(x1, x2, params={'p':.5}):
    return gamma(params['p']*(x1+x2-2.) + 1.) / (gamma(x1)*gamma(x2))**params['p']

def composite_rbf(x1, x2, params={'t':.09, 'b':1.5}):
    #if x1 > 1: f1 = np.exp(1) - 1
    #else: f1 = np.exp(x1/params['b']) - 1
    #if x2 > 1: f2 = np.exp(1) - 1
    #else: f2 = np.exp(x2/params['b']) - 1
    return .08*x1*rbf(x1,x2,params)*x2

def gamma_kld(a1, a2, theta1=1., theta2=1.):
    return (a1-a2)*polygamma(0,a1) - np.log(gamma(a1)) + np.log(gamma(a2)) + a2*(np.log(theta2)-np.log(theta1)) + a1*((theta1-theta2)/theta2)


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
