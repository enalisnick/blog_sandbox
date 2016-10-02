import matplotlib
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from kernels import *

x = np.linspace(0, 20, 1000)
y = .1

rbf_arr = []
diff_arr = []
fish_arr = []
prob_arr = []
comp_arr = []
kld_arr = []
for idx in xrange(len(x)):
    rbf_arr.append(rbf(np.log(x[idx]), np.log(y)))
    diff_arr.append(gamma_diffusion(x[idx],y))
    fish_arr.append(gamma_fisher(x[idx],y))
    prob_arr.append(gamma_prob(x[idx],y))
    comp_arr.append(composite_rbf(x[idx],y))
    kld_arr.append(kld(x[idx],y, gamma_kld))

plt.plot(x, rbf_arr, 'b-', linewidth=4, label="RBF (log space)")
plt.plot(x, diff_arr, 'r-', linewidth=4, label="Info. Diffusion")
plt.plot(x, fish_arr, 'g-', linewidth=4, label="Fisher")
plt.plot(x, prob_arr, 'k-', linewidth=4, label="Probability Prod.")
plt.plot(x, comp_arr, 'y-', linewidth=4, label="Composite")
plt.plot(x, kld_arr, 'c-', linewidth=4, label="KLD")
plt.plot([y]*1000, np.linspace(0, 2, 1000), 'k--', linewidth=3, alpha=.5)
plt.xlim([0,20])
#plt.ylim([0,1.5])
plt.xlabel(r"$x_{1}$ Values")
plt.ylabel(r"$K(x_{1},x_{2})$")
plt.title(r"$K(x_{1},x_{2})$ Values for $x_{2}=%.1f$"%(y))
plt.legend()
plt.show()
