import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from kernels import *
from scipy.stats import norm

x = np.linspace(0, 1, 1000)
y = .5

rbf_arr = []
fish_arr = []
prob_arr = []
sobolev_arr = []
kld_arr = []
for idx in xrange(len(x)):
    rbf_arr.append(rbf(norm.ppf(x[idx]), norm.ppf(y)))
    fish_arr.append(beta_fisher(x[idx],y))
    prob_arr.append(bernoulli_prob_prod(x[idx],y))
    #comp_arr.append(composite_rbf(x[idx],y))
    sobolev_arr.append(unit_sobolev(x[idx], y))
    kld_arr.append(kld(x[idx],y, bernoulli_kld))

plt.plot(x, rbf_arr, 'b-', linewidth=4, label="RBF (probit transform)")
plt.plot(x, sobolev_arr, 'r-', linewidth=4, label="Sobolev")
plt.plot(x, fish_arr, 'g-', linewidth=4, label="Fisher")
plt.plot(x, prob_arr, 'k-', linewidth=4, label="Probability Prod.")
#plt.plot(x, comp_arr, 'y-', linewidth=4, label="Composite")
plt.plot(x, kld_arr, 'c-', linewidth=4, label="KLD")
plt.plot([y]*1000, np.linspace(0, 1, 1000), 'k--', linewidth=3, alpha=.5)
plt.xlim([0,1])
plt.ylim([0,1.1])
plt.xlabel(r"$x_{1}$ Values")
plt.ylabel(r"$K(x_{1},x_{2})$")
plt.title(r"$K(x_{1},x_{2})$ Values for $x_{2}=%.1f$"%(y))
plt.legend()
plt.show()
