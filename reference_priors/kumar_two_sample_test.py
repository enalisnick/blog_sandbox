import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from autograd.scipy.special import polygamma
from autograd.numpy.linalg import inv
import cPickle as cp

def sample_from_kumaraswamy(a,b):
    return ( 1. - np.random.uniform(.001,.999)**(1./b) )**(1./a)

def sample_from_beta(alpha, beta):
    return np.random.beta(alpha, beta)

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

def compute_mmd(samples_p, samples_q):
    term1 = 0.
    term2 = 0.
    term3 = 0.
    kernel = beta_fisher

    for i_idx in xrange(len(samples_p)):
        for j_idx in xrange(len(samples_q)):
            term1 += kernel(samples_p[i_idx], samples_p[j_idx])
            term2 += kernel(samples_p[i_idx], samples_q[j_idx])
            term3 += kernel(samples_q[i_idx], samples_q[j_idx])
        
    normalizer = 1./(len(samples_p)**2)
    
    return np.sqrt( normalizer * (term1 - 2*term2 + term3 ) )

if __name__ == '__main__':
    
    # kumar params
    a = .37
    b = .44

    # beta params
    alpha = .5
    beta = .5

    N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100] # [x for x in xrange(50,1050,50)] # number of samples
    resamples = 10

    mu = []
    std = []
    all_MMDs = []
    for n in N:
        print "Current sample size: %d" %(n)
        MMDs = []
        all_MMDs.append([])
        for resample_idx in xrange(resamples):
            kumar_samples = [sample_from_kumaraswamy(a,b) for i in xrange(n)]
            beta_samples = [sample_from_beta(alpha,beta) for i in xrange(n)]
            mmd = compute_mmd(kumar_samples, beta_samples)
            MMDs.append(mmd)
            all_MMDs[-1].append(mmd)

        mu.append(np.mean(MMDs))
        std.append(np.std(MMDs))
    
    cp.dump({'all':all_MMDs, 'mu':mu, 'std':std}, open('beta_vs_kumar_MMDs.pkl','wb'))

    plt.figure()
    plt.errorbar(N, mu, yerr=std, fmt='kx-', markersize=10, mew=5, linewidth=3, label="Parametric")
    plt.title("Maximum Mean Discrepancy as Sample Size Increases")
    plt.xlabel(r"Sample Size")
    plt.ylabel(r"Maximum Mean Discrepancy (MMD)")
    plt.legend()
    plt.show()

