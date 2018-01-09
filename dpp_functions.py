import numpy as np
from scipy.linalg import orth
from numpy.linalg import eig


def elem_sympoly(vals, k):
    """Uses Newton's identities to compute elementary symmetric polynomials."""
    N = vals.shape[0]
    E = np.zeros([k+1, N+1])
    E[0,] = 1
    
    for i in range(1, k+1):
        for j in range(1, N+1):
            E[i,j] = E[i, j-1] + vals[j-1] * E[i-1, j-1]
    
    return E[:,1:]


def sample_k(vals, k):
    """
    """
    N = vals.shape[0]
    E = elem_sympoly(vals, k)[:,1:]
    sample = np.zeros(N, dtype=int)
    rem = k

    for elem, val in reversed(list(enumerate(vals))):

        # Check if we chose k elements
        if not rem:
            break
            
        # Compute conditional marginal of elem
        marg = val * E[rem-1, elem-1] / E[rem, elem]

        # Sample elem
        if np.random.rand() < marg:
            sample[elem] = 1
            rem -= 1
        
        # Check if all remaining elements will be chosen
        if elem == rem:
            sample[np.arange(rem)] = 1
            break

    return sample


def sample_from_dpp(item_vectors, k=None):
    """
    This function expects... 
    
    Arguments 
    item_vectors: NumPy 2D Array of N items X D dimensions
    k: Number of samples to return for k-DPP

    Returns
    sample: List containing indicies of selected items

    """
    n = item_vectors.shape[0] # number of items in ground set
    gram_mat = np.dot(item_vectors, item_vectors.T)
    vals, vecs = eig(gram_mat)
    
    # k-DPP
    if k > 0:
        index = sample_k(vals, k) # sample_k, need to return index

    # Sample set size
    else:
        index = (np.random.rand(n) < (vals / (vals + 1)))
        k = np.sum(index)
    
    # Check for empty set
    if k == 0:
        return [] #np.empty(0)
    
    # Check for full set
    if k == n:
        return np.arange(k, dtype=float) 
    
    V = vecs[:, index]

    # Sample a set of k items 
    items = list()

    for i in range(k):
        p = np.sum(V**2, axis=1)
        p = np.cumsum(p / np.sum(p)) # item cumulative probabilities
        item = (np.random.rand() <= p).argmax()
        items.append(item)
        
        # Delete one eigenvector not orthogonal to e_item and find new basis
        j = (np.abs(V[item, :]) > 0).argmax() 
        Vj = V[:, j]
        V = orth(V - (np.outer(Vj,(V[item, :] / Vj[item])))) 
    
    items.sort()
    sample = items #np.array(items, dtype=float)    
    
    return sample 
