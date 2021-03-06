# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
from math_utils import safe_log
from multiprocessing import Pool    
    
def expected_theta(v_params, m_params, doc, counts):
    nsamples = 1000

    #Precalculate some values for the sampling loop
    t1 = 0.5 * np.linalg.slogdet(m_params.inv_sigma)[1]
    t1 -= 0.5 * len(m_params.beta) * np.log(2 * np.pi)
    
    betaDoc = m_params.beta[:, doc]
    sigma = np.diag(np.sqrt(v_params.nu_sq))
    
    samples = []
    
    for _ in xrange(nsamples):
        eta = np.random.multivariate_normal(v_params.lambd, sigma)
        
        theta = np.exp(eta)
        theta /= sum(theta)
        
        eta_mu = eta - m_params.mu
        
        t = t1 - 0.5 * eta_mu.dot(m_params.inv_sigma.dot(eta_mu)) +\
            counts.dot(safe_log(theta.dot(betaDoc)))
        t2 = np.sum(safe_log(scipy.stats.multivariate_normal.pdf(eta - v_params.lambd, np.zeros(len(eta)), sigma)))
            
        samples.append(t - t2 + safe_log(theta))
    
    e_theta = scipy.misc.logsumexp(samples, axis=0) - np.log(nsamples)
    norm = scipy.misc.logsumexp(e_theta)
    return np.exp(e_theta - norm)
    
class ETWorker:
    def __init__(self, m_params):
        self.m_params = m_params
    def __call__(self, x):
        return expected_theta(x[0], self.m_params, x[1], x[2])

def parallel_expected_theta(v_params_list, m_params, doc_words, doc_counts):
    pool = Pool(processes=2)
    result = pool.map(ETWorker(m_params), zip(v_params_list, doc_words, doc_counts))
    pool.close()
    return result