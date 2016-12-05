import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


def logistic(x):
    return 1./(1+np.exp(-x))

def softplus(x):
    return np.log(1+np.exp(x))

def kumaraswamyPDF(x, a,b):
    return a*b * (x**(a-1)) * (1-(x**a))**(b-1)

def compute_nonParam_ref_approx(t=50, m=200, s=500, lr=0.0001, rs=1000):
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


def compute_param_ref_approx(t=25, m=200, s=500, lr=0.0001):
    a_w = np.random.normal()
    b_w = np.random.normal()

    for t_idx in xrange(t):
        ### term 1                                                                                                                                                      
        # compute p_hat                                                                                                         
        epsilon = np.random.uniform()
        a = softplus(a_w)
        b = softplus(b_w)
        theta_hat = (1.-(epsilon**(1./b)))**(1./a)

        # sample data                                                                                                                                       
        Us = np.random.uniform(0,1,size=(1,m))
        Ls = np.log(Us) - np.log(1-Us)
        alpha = -np.log(1-theta_hat)
        data = logistic(Ls + np.log(alpha))

        ### term2                                                                                                                                                       
        # compute p_hat k's                                                                                                                                            
        epsilon_ks = [np.random.uniform() for s_idx in xrange(s)]
        theta_hat_ks = [ (1.-(epsilon_k**(1./b)))**(1./a) for epsilon_k in epsilon_ks]

        ### update params                                                                                                                                               
        # term1                                                                                                                                                        
        dData_dp = np.sum(data * (1-data)) * (-1./((1-theta_hat)*np.log(1-theta_hat)))
        dTerm1_dp = dData_dp*np.log(theta_hat) + np.sum(data)/theta_hat + (m - dData_dp)*np.log(1-theta_hat) + -(m - np.sum(data))/(1-theta_hat)

        # term2                                                                                                                                         
        dTerm2_dp = [dData_dp*np.log(theta_k) + np.sum(data)/theta_k + (m - dData_dp)*np.log(1-theta_k) + -(m - np.sum(data))/(1-theta_k) for theta_k in theta_hat_ks]

        dp0_da = theta_hat * np.log(1. - (epsilon**(1./b))) * (-1./(a**2))
        dp0_db = (1./a) * (1. - (epsilon**(1./b)))**(1./a - 1.) * -(epsilon**(1./b)) * np.log(epsilon) * (-1./(b**2))
        dpk_da = [theta_hat_ks[k] * np.log(1. - (epsilon_ks[k]**(1./b))) * (-1./(a**2)) for k in xrange(s)]
        dpk_db = [(1./a) * (1. - (epsilon_ks[k]**(1./b)))**(1./a - 1.) * -(epsilon_ks[k]**(1./b)) * np.log(epsilon_ks[k]) * (-1./(b**2)) for k in xrange(s)]

        a_w += lr * (dTerm1_dp*dp0_da*logistic(a_w) - np.mean([dTerm2_dp[k]*dpk_da[k]*logistic(a_w) for k in xrange(s)]))
        b_w += lr * (dTerm1_dp*dp0_db*logistic(b_w) - np.mean([dTerm2_dp[k]*dpk_db[k]*logistic(b_w) for k in xrange(s)]))

    return softplus(a_w), softplus(b_w)


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
    y_approxRef = compute_nonParam_ref_approx(rs=1000)
    kumar_a, kumar_b = compute_param_ref_approx()
    y_param_approxRef = kumaraswamyPDF(x, kumar_a, kumar_b)
    
    plt.figure()
    plt.plot(x, y, 'k--', linewidth=5, label="Jeffreys Prior Beta(.5,.5)")
    plt.plot(x, y_param_approxRef, 'g-', linewidth=5, label=r"Param Approx, Kumaraswamy($a=$%.2f, $b=$%.2f)"%(kumar_a,kumar_b))
    plt.plot(x_berger, y_berger, 'rx--', markersize=10, mew=5, linewidth=3, label="Brute Force Algorithm")
    n, bins, patches = plt.hist(y_approxRef, facecolor="b", normed=True, stacked=True, alpha=.6, bins=np.arange(0, 1.1, .05), edgecolor="none", label="Non-Param Approx, Linear Model")
    plt.xlim([0,1])
    plt.ylim([0,5])
    plt.title("Reference Priors for Bernoulli Parameter")
    plt.xlabel(r"$p$")
    plt.ylabel(r"$p^{*}(p)$")
    plt.legend()
    plt.show()

