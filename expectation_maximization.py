# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 18:02:06 2014

@author: mildbyte
"""

import numpy as np
from variational_inference import variational_inference, likelihood_bound
from math_utils import cor_mat

def dsm_rmse(inf, ref):
    return np.sqrt(np.sum(np.square(inf-ref)) / ref.size) / (np.max(ref) - np.min(ref))


class Model():
    def __init__(self, mu, sigma, beta):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        
        self.inv_sigma = np.linalg.inv(sigma)
    def __str__(self):
        return "mu: " + str(self.mu) + "; sigma: " + str(self.sigma) + "; beta: " + str(self.beta)
        
"""Trains the model on the corpus with given pathway priors and returns the MLE for sigma, mu and beta."""
def expectation_maximization(corpus, word_counts, no_pathways, pathway_priors, max_iterations=5, initial_params=None):
    if initial_params is not None:
        m_params = initial_params
    else:
        m_params = Model(np.zeros(no_pathways), np.identity(no_pathways), np.array(pathway_priors))
    
    expanded_counts = []
    for w, c in zip(corpus, word_counts):
        exp_count = np.zeros(pathway_priors.shape[1])
        exp_count[w] = c
        expanded_counts.append(exp_count)
    
    iteration = 0
    
    while True:
        print "Iteration: " + str(iteration)
            
        print "Performing variational inference..."

        #This is to avoid writing a safe_log function that wastes
        #a lot of time because it clips all values in the array to > 0.0001.
        #Has the same effect: log values can't be negative infinity anymore.
        #Could introduce a negligible bias (because VI gets phi from beta),
        #but since we reset these positions (updated from phi) to 1e-100 every
        #time, it shouldn't accumulate.
        beta_zeros = m_params.beta == 0
        m_params.beta[beta_zeros] = 1e-100
        
        params = [None] * len(corpus)

        #Can pass previous v_params to speed up convergence
        for i, d, c in zip(xrange(len(corpus)), corpus, word_counts):
            params[i] = variational_inference(d, c, m_params, params[i])
        
        old_l_bound = sum([likelihood_bound(p, m_params, d, c, sum(c)) for (p, d, c) in zip(params, corpus, word_counts)])
        print "Old bound: %.2f" % old_l_bound
        
        mu = np.sum([p.lambd for p in params], axis=0) / len(corpus)
        sigma = np.sum([np.diag(p.nu_sq) + np.outer(p.lambd, p.lambd) for p in params], axis=0)
        sigma /= len(corpus)
        sigma -= np.outer(mu, mu)        
        sigma += np.eye(len(sigma)) * 0.00001 #Ridge on the principal diagonal to avoid singular matrices
        
        beta = np.zeros(m_params.beta.shape)
        
        for param, words, count in zip(params, corpus, expanded_counts):
            exp_phi = np.zeros(pathway_priors.shape).T
            exp_phi[words] = param.phi #Sparse phi s.t. exp_phi[w, i] = phi[doc^-1[w], i] or 0 if doc^-1 is undefined at w
            beta += np.multiply(count, exp_phi.T)
        
        beta /= np.sum(beta, axis=1)[:, np.newaxis]
#        
#        beta = np.zeros(m_params.beta.shape)
#        for d in xrange(len(corpus)):
#            for w in xrange(len(corpus[d])):
#                word = corpus[d][w]
#                count = word_counts[d][w]
#                for i in xrange(len(beta)):
#                    beta[i, word] += count * params[d].phi[w, i]
#        
#        for i in xrange(len(beta)):
#            beta[i] /= np.sum(beta[i])
        
        
        
        m_params = Model(mu, sigma, beta)
        iteration += 1

        
        new_l_bound = sum([likelihood_bound(p, m_params, d, c, sum(c)) for (p, d, c) in zip(params, corpus, word_counts)])
        delta = abs((new_l_bound - old_l_bound)/old_l_bound)
        print "New bound: %.2f, difference: %.6f" % (new_l_bound, delta)
        
        if (delta < 1e-5 or iteration >= max_iterations):
            break
        
    m_params.beta[beta_zeros] = 0
    return m_params, params