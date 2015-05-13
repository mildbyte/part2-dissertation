# -*- coding: utf-8 -*-

import numpy as np
        
def safe_log(x):
    return np.log(np.clip(x, a_min=1e-6, a_max=np.inf))

def cor_mat(sigma):
    inv_sigma = np.linalg.inv(sigma)
    result = np.zeros(sigma.shape)
    for i in xrange(inv_sigma.shape[0]):
        for j in xrange(inv_sigma.shape[1]):
            result[i, j] = np.abs(inv_sigma[i, j])/np.sqrt(inv_sigma[i,i]*inv_sigma[j,j])
    return result