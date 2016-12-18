import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta
from autograd.scipy.special import polygamma
from autograd.numpy.linalg import inv
import cPickle as cp


if __name__ == '__main__':
    
    kumar = cp.load(open("beta_vs_kumar_MMDs.pkl","rb"))
    berger = cp.load(open("beta_vs_berger_MMDs.pkl","rb"))

    N = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    plt.figure()
    plt.errorbar(N, kumar['mu'], yerr=kumar['std'], fmt='rx-', markersize=10, mew=5, linewidth=3, label="Parametric, Kumaraswamy")
    plt.errorbar(N, berger['mu'], yerr=berger['std'], fmt='bx-', markersize=10, mew=5, linewidth=3, label="Brute-Force Alg (Berger et al.)")
    plt.title("Maximum Mean Discrepancy as Sample Size Increases")
    plt.xlabel(r"Sample Size")
    plt.ylabel(r"Maximum Mean Discrepancy (MMD)")
    plt.xlim([N[0]-1.5, N[-1]+1.5])
    plt.legend()
    plt.show()

