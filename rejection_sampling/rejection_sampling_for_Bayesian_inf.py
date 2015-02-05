#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")
        norm_const = 1.0/ ( np.power((2*np.pi),float(size)/2) * np.power(det,1.0/2) )
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)
        result = np.power(np.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

# get number of samples to draw from posterior
try:
    num_of_samples = int(sys.argv[1])
except:
    num_of_samples = 1000

# load Waldo data
waldo_locations = []
f = open('../data/wheres-waldo-locations.csv', 'rU')
csv_f = csv.reader(f)
for row in csv_f:
    row[-1] = row[-1].rstrip()
    waldo_locations.append([float(row[2]), float(row[3])])
waldo_locations = np.matrix(waldo_locations)

# calculate Gaussian MLE
mle_mu = np.mean(waldo_locations, axis=0)
mle_covar = np.zeros((2,2))
for point in waldo_locations:
    mle_covar += np.dot((point - mle_mu).T,(point - mle_mu))
mle_covar = 1./waldo_locations.shape[0] * mle_covar

# calculate likelihood under MLE
mle_likelihood = 1.0
for point in waldo_locations:
    mle_likelihood *= norm_pdf_multivariate(np.array(point)[0], mu=np.array(mle_mu)[0], sigma=mle_covar)
mle_likelihood = mle_likelihood[0,0]

# iterate until number of desired samples is reached
while num_of_samples > 0:
    # draw a sample from the prior
    sample = np.random.laplace(loc=mle_mu, scale=2.0, size=1)
    numerator =


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