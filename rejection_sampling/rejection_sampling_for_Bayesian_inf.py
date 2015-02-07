#!/usr/bin/python
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys

def multivariate_gauss_pdf(x, mu, sigma):
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

def multivariate_cauchy_transform(mu, sigma, deg_of_freedom):
    return np.random.multivariate_normal([0,0], sigma)*np.sqrt(deg_of_freedom/np.random.chisquare(deg_of_freedom))+mu

def compute_likelihood_under_gauss(data, mu, sigma, product_flag=1):
    mle_likelihood = 1.0
    likelihood_per_point = []
    for point in data:
        l = multivariate_gauss_pdf(np.array(point)[0], mu=np.array(mu)[0], sigma=sigma)
        mle_likelihood *= l
        likelihood_per_point.append(l)
    if product_flag:
        return mle_likelihood[0,0]
    else:
        return likelihood_per_point


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
max_likelihood = compute_likelihood_under_gauss(waldo_locations, mle_mu, mle_covar)

# define prior a mu -- Cauchy
degrees_of_freedom = 1 # function generalizes to other student-t's
prior_sigma = mle_covar
prior_mu = mle_mu

accepted_samples = []
rejected_samples = []
# iterate until number of desired samples is reached
while num_of_samples > 0:
    # draw a sample from the prior
    sample_from_q = multivariate_cauchy_transform(np.matrix(prior_mu), np.matrix(prior_sigma), degrees_of_freedom)
    # draw height
    sample_from_uniform = np.random.uniform(0,1)
    # compute ratio
    weight = compute_likelihood_under_gauss(waldo_locations, sample_from_q, np.matrix([[3,0],[0,3]])) / max_likelihood
    print weight
    # accept or reject
    if sample_from_uniform <= weight:
        accepted_samples.append(sample_from_q)
        num_of_samples -= 1
    else:
        rejected_samples.append(sample_from_q)

# plot various graphs
plt.figure(1)

##### Cauchy Prior #####
plt.subplot(2,2,1)
plt.title('Cauchy Prior')
plt.scatter(accepted_samples+rejected_samples, 'ok')
plt.xlim([0,13])
plt.ylim([0,8])

##### Max Likelihood #####
plt.subplot(2,2,2)
plt.title('Maximum Likelihood Gaussian')
plt.scatter(waldo_locations, 'ok')
xi = np.linspace(0,13,1000)
yi = np.linspace(0,8,1000)
z = compute_likelihood_under_gauss(np.hstack(xi.T, yi.T), mle_mu, mle_covar)
## grid the data.
zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
levels = [0.2, 0.4, 0.6, 0.8, 1.0]
# contour the gridded data, plotting dots at the randomly spaced data points.
CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
#CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
plt.colorbar() # draw colorbar
plt.xlim([0,13])
plt.ylim([0,8])

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()