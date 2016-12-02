import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta

def bergerAlg(K=10, k2=500, M=50, J=100):
    prior = []
    for k_idx in xrange(K):
        theta = np.random.uniform(0.0001,.999)
        log_posts = []

        for j_idx in xrange(J):
            data = [np.random.binomial(n=1, p=theta) for m_idx in xrange(M)]
            
            # denominator
            p_x = []
            for theta2 in np.linspace(0.0001,.999,k2):
                p = 1.
                for d in data:
                    p *= theta2**d * (1-theta2)**(d-1)
                p_x.append(p)
            p_x = np.array(p_x).sum()
                
            # numerator
            p = 1.
            for d in data:
                p *= theta**d * (1-theta)**(d-1)
            
            log_posts.append(np.log(p) - np.log(p_x))
            
        log_posts = np.array(log_posts)
        prior.append( (theta, np.exp(log_posts.mean())) )
    return prior


if __name__ == '__main__':
    
    x = np.linspace(0,1,1000)
    y = beta.pdf(x, .5, .5)

    p = bergerAlg()
    x_berger = []
    y_berger = []
    for a,b in p:
        x_berger.append(a)
        y_berger.append(b)

    print y_berger
    #x_berger, y_berger = zip(*sorted(zip(x_berger, y_berger)))
    
    plt.figure()
    plt.plot(x, y, 'b-', linewidth=5)
    plt.scatter(x_berger, y_berger) #, 'r--', linewidth=5)
    plt.xlim([0,1])
    plt.ylim([0,5])
    plt.show()

