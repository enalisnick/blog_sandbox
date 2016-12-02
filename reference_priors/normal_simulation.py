import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def softplus(x):
    return np.log(1+np.exp(x))

def logistic(x):
    return 1./(1+np.exp(-x))

def gaussJeffPrior(x):
    return 1./x

def compute_ref_approx(t=0, s=100, lr=0.1, rs=1000):
    w = np.random.normal(size=(10,1))
    b = 0.
    for t_idx in xrange(t):
        # term 1
        epsilon = np.random.normal(size=(1,10))
        a = np.dot(epsilon,w) + b
        theta_hat = softplus(a)

        # term2
        epsilon_ks = [np.random.normal(size=(1,10)) for s_idx in xrange(s)]
        a_ks = [np.dot(epsilon_k,w)+b for epsilon_k in epsilon_ks]
        theta_hat_ks = [softplus(a_k) for a_k in a_ks]
        
        # update params
        neg_w_grad = [(-1./theta_hat_ks[k] * logistic(a_ks[k]) * epsilon_ks[k].T) for k in xrange(s)]
        w += lr * (-1./theta_hat * logistic(a) * epsilon.T + np.mean(neg_w_grad))

        neg_b_grad = [(-1./theta_hat_ks[k] * logistic(a_ks[k]) * 1) for k in xrange(s)]
        b += lr * (-1./theta_hat * logistic(a) * 1 + np.mean(neg_b_grad))

    return [softplus(np.dot(np.random.normal(size=(1,10)),w)+b) for k in xrange(rs)]


def bergerAlg(K=10, k2=500, M=500, J=100):
    prior_p = []
    prior_v = []

    thetas = np.linspace(0.0001, 5, K)
    for theta in thetas:
        log_posts = []

        for j_idx in xrange(J):
            data = np.random.normal(loc=0., scale=theta, size=(1,M))

            # denominator                                                                                                  
            p_x = 0.
            for theta2 in np.linspace(0.0001, .999, k2):
                p_x += norm.pdf(data, scale=theta2)

            log_posts.append( np.log(norm.pdf(data,scale=theta), p_x) )

        log_posts = np.array(log_posts)
        prior_p.append(theta)
        prior_v.append(np.exp(log_posts.mean()))
    return prior_p, prior_v




if __name__=='__main__':
    
    x = np.linspace(.00001, 5, 500)
    
    y_jeff = gaussJeffPrior(x)
    y_approxRef = compute_ref_approx(rs=1000)
    x_berger, y_berger = bergerAlg()
    
    sigma_bins = [z for z in np.linspace(0,10,20)]

    plt.figure()
    plt.plot(x, y_jeff, 'b-', linewidth=5, label=r"Jeffreys Prior $1/\sigma$")
    plt.plot(x_berger, y_berger, 'rx--', markersize=10, mew=5, linewidth=5, label="Brute Force Algorithm")
    n, bins, patches = plt.hist(y_approxRef, facecolor="k", normed=True, stacked=True, alpha=.5, bins=np.arange(0, 5.1, .1), edgecolor="none", label="Linear Model Approx.")
    plt.ylim(-.01,6)
    plt.xlim(0,5)
    plt.legend()
    plt.title("Reference Priors for Gaussian Scale Parameter")
    plt.ylabel(r"$p^{*}(\sigma)$")
    plt.xlabel(r"$\sigma$")
    plt.show()
