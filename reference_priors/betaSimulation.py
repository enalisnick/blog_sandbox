import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def logistic(x):
    return 1./(1+np.exp(-x))

def compute_ref_approx(t=10, m=100, s=100, lr=0.01, rs=1000):
    w = np.random.normal(size=(10,1))
    b = 0.
    for t_idx in xrange(t):
        ### term 1

        # compute p_hat
        epsilon = np.random.normal(size=(1,10))
        a = np.dot(epsilon,w) + b
        theta_hat = logistic(a)

        # sample data
        Us = np.random.uniform(0,1,size=(1,m))
        Ls = np.log(Us) - np.log(1-Us)
        alpha = -np.log(1-theta_hat)
        data = logistic(Ls + np.log(alpha))

        ### term2                                                                                                                   
        # compute p_hat k's
        epsilon_ks = [np.random.normal(size=(1,10)) for s_idx in xrange(s)]
        a_ks = [np.dot(epsilon_k,w)+b for epsilon_k in epsilon_ks]
        theta_hat_ks = [logistic(a_k) for a_k in a_ks]

        ### update params   

        # term1
        dData_dp = np.sum(data * (1-data)) * (-1./((1-theta_hat)*np.log(1-theta_hat)))
        dTerm1_dp = dData_dp*np.log(theta_hat) + np.sum(data)/theta_hat + (m - dData_dp)*np.log(1-theta_hat) + -(m - np.sum(data))/(1-theta_hat)

        # term2
        dTerm2_dp = [dData_dp*np.log(theta_k) + np.sum(data)/theta_k + (m - dData_dp)*np.log(1-theta_k) + -(m - np.sum(data))/(1-theta_k) for theta_k in theta_hat_ks]
        
        dp0_db = theta_hat*(1-theta_hat)
        dp0_dw = dp0_db * epsilon.T
        dpk_db = [theta_hat_ks[k]*(1-theta_hat_ks[k]) for k in xrange(s)]
        dpk_dw = [dpk_db[k]*epsilon_ks[k] for k in xrange(s)]

        w += lr * (dTerm1_dp*dp0_dw - np.mean([dTerm2_dp[k]*dpk_dw[k] for k in xrange(s)]))
        b += lr * (dTerm1_dp*dp0_db - np.mean([dTerm2_dp[k]*dpk_db[k] for k in xrange(s)]))

    return [logistic(np.dot(np.random.normal(size=(1,10)),w)+b)[0,0] for k in xrange(rs)]


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
    y_approxRef = compute_ref_approx(rs=1000)
    
    plt.figure()
    plt.plot(x, y, 'b-', linewidth=5, label="Jeffreys Prior, Beta(.5,.5)")
    plt.plot(x_berger, y_berger, 'rx--', markersize=10, mew=5, linewidth=5, label="Brute Force Algorithm")
    n, bins, patches = plt.hist(y_approxRef, facecolor="k", normed=True, stacked=True, alpha=.5, bins=np.arange(0, 1.1, .05), edgecolor="none", label="Linear Model Approx.")
    plt.xlim([0,1])
    plt.ylim([0,5])
    plt.title("Reference Priors for Bernoulli Parameter")
    plt.xlabel(r"Bernoulli Parameter $p$")
    plt.ylabel(r"$p^{*}(p)$")
    plt.legend()
    plt.show()

