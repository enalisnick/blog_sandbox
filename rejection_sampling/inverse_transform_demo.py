#!/usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import sys

# get number of samples to draw
try:
    int(sys.argv[1])
    num_of_samples = int(sys.argv[1])
except:
    num_of_samples = 1000

# sample from distributions directly
uniform_samples = np.random.uniform(0,1,num_of_samples)
direct_cauchy_samples = np.random.standard_cauchy(num_of_samples)
direct_beta_samples = np.random.beta(0.5, 0.5, num_of_samples)
# compute inverse cdf transforms
transformed_cauchy_samples = np.tan(np.pi*(uniform_samples - 0.5))
transformed_beta_samples = np.power(np.sin(np.pi/2 * uniform_samples),2)

# plot histograms to compare direct vs transformed
cauchy_bins = []
beta_bins = []
for x in xrange(40):
    cauchy_bins.append(x-20)
for x in xrange(100):
    beta_bins.append(x*10**(-2))
plt.figure(1)

##### Cauchy Subplots #####
plt.subplot(2,2,1)
plt.title('Cauchy Direct Samples')
n, bins, patches = plt.hist(direct_cauchy_samples, cauchy_bins, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'k', 'alpha', 0.75)
plt.xlim([-20,20])

plt.subplot(2,2,2)
plt.title('Transformed Uniform')
n, bins, patches = plt.hist(transformed_cauchy_samples, cauchy_bins, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
plt.xlim([-20,20])

##### Beta Subplots #####
plt.subplot(2,2,3)
plt.title('Beta Direct Samples')
n, bins, patches = plt.hist(direct_beta_samples, beta_bins, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'k', 'alpha', 0.75)
plt.xlim([0,1])

plt.subplot(2,2,4)
plt.title('Transformed Uniform')
n, bins, patches = plt.hist(transformed_beta_samples, beta_bins, normed=1, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'b', 'alpha', 0.75)
plt.xlim([0,1])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()