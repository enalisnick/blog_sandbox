import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def logistic(x):
    return 1./(1+np.exp(-x))

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

    thetas = np.linspace(0, 1, K)
    for theta in thetas:
        log_posts = []

        for j_idx in xrange(J):
            data = np.random.binomial(n=1, p=theta, size=(1,M)) 
            T = data.sum()

            # denominator
            p_x = 0.
            for theta2 in np.linspace(0.0001,.999,k2):
                p_x += theta2**T * theta2**(M-T)
                
            log_posts.append( np.log(theta**T * theta2**(M-T) / p_x) )
            
        log_posts = np.array(log_posts)
        prior_p.append(theta)
        prior_v.append(np.exp(log_posts.mean()) * 2 + .5)
    return prior_p, prior_v


if __name__ == '__main__':
    
    x = np.linspace(0,1,1000)

    y = beta.pdf(x, .5, .5)
    x_berger, y_berger = bergerAlg()
    
    plt.figure()
    plt.plot(x, y, 'b-', linewidth=5, label="Jeffreys Prior, Beta(.5,.5)")
    plt.plot(x_berger, y_berger, 'rx--', markersize=10, mew=5, linewidth=5, label="Brute Force Algorithm")
    plt.xlim([0,1])
    plt.ylim([0,5])
    plt.title("Reference Priors for Bernoulli Parameter")
    plt.xlabel(r"Bernoulli Parameter $p$")
    plt.ylabel(r"$p^{*}(p)$")
    plt.legend()
    plt.show()

