# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 18:04:20 2014

@author: mildbyte
"""

import numpy as np
import scipy.stats
from math_utils import safe_log
from multiprocessing import Pool    
    

"""Sampling of the likelihood based on the variational posterior"""
def sample_term(v_params, m_params, doc, counts, eta, t1):
    theta = np.exp(eta)
    theta /= sum(theta)
    
    t1 += counts.dot(safe_log(theta.dot(m_params.beta[:, doc])))
#    
#    for (n, c) in zip(xrange(len(doc)), counts):
#        t1 += c * safe_log(np.sum(theta * m_params.beta[:, doc[n]]))
#        
    t2 = np.sum(safe_log(scipy.stats.multivariate_normal.pdf(eta - v_params.lambd, np.zeros(len(eta)), np.diag(np.sqrt(v_params.nu_sq)))))
    
    return t1 - t2

def sample_lhood(v_params, m_params, doc, counts):
    nsamples = 10000

    t1 = 0.5 * np.linalg.slogdet(m_params.inv_sigma)[1]
    t1 -= 0.5 * (np.diag(v_params.nu_sq).dot(m_params.inv_sigma)).trace()
    lambda_mu = v_params.lambd - m_params.mu
    t1 -= 0.5 * lambda_mu.dot(m_params.inv_sigma.dot(lambda_mu))
    
    def terms():
        for _ in xrange(nsamples):
            eta = np.random.multivariate_normal(v_params.lambd, np.diag(np.sqrt(v_params.nu_sq)))
            yield sample_term(v_params, m_params, doc, counts, eta, t1)
            
    return scipy.misc.logsumexp(np.array(terms())) - safe_log(nsamples)
    
def expected_theta(v_params, m_params, doc, counts):
    nsamples = 1000

    t1 = 0.5 * np.linalg.slogdet(m_params.inv_sigma)[1]
    t1 -= 0.5 * (np.diag(v_params.nu_sq).dot(m_params.inv_sigma)).trace()
    lambda_mu = v_params.lambd - m_params.mu
    t1 -= 0.5 * lambda_mu.dot(m_params.inv_sigma.dot(lambda_mu))
    
    def terms():
            
        for _ in xrange(nsamples):
            eta = np.random.multivariate_normal(v_params.lambd, np.diag(np.sqrt(v_params.nu_sq)))
            
            theta = np.exp(eta)
            theta /= sum(theta)
        
            w = sample_term(v_params, m_params, doc, counts, eta, t1)
            
            yield w + safe_log(theta)
    
    samples = list(terms())

    e_theta = scipy.misc.logsumexp(samples, axis=0) - safe_log(nsamples)
    norm = scipy.misc.logsumexp(e_theta)
    return np.exp(e_theta - norm)
    
class ETWorker:
    def __init__(self, m_params):
        self.m_params = m_params
    def __call__(self, x):
        return expected_theta(x[0], self.m_params, x[1], x[2])

def parallel_expected_theta(v_params_list, m_params, doc_words, doc_counts):
    pool = Pool(processes=8)
    result = pool.map(ETWorker(m_params), zip(v_params_list, doc_words, doc_counts))
    pool.close()    
    return result