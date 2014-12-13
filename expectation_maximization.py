# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 18:02:06 2014

@author: mildbyte
"""

import numpy as np
from variational_inference import variational_inference, likelihood_bound
from multiprocessing import Pool


class Model():
    def __init__(self, mu, sigma, beta):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        
        self.inv_sigma = np.linalg.inv(sigma)
    def __str__(self):
        return "mu: " + str(self.mu) + "; sigma: " + str(self.sigma) + "; beta: " + str(self.beta)


class VIWorker:
    def __init__(self, m_params):
        self.m_params = m_params
    def __call__(self, x):
        return variational_inference(x[0], x[1], self.m_params)   
        
"""Trains the model on the corpus with given pathway priors and returns the MLE for sigma, mu and beta."""
def expectation_maximization(corpus, word_counts, no_pathways, pathway_priors, max_iterations=10):
    m_params = Model(np.zeros(no_pathways), np.identity(no_pathways), np.array(pathway_priors))
    
    iteration = 0
    
    #pool = Pool(processes=8)
    
    while True:
        print "Iteration: " + str(iteration)
            
        print "Performing variational inference..."

        #params = pool.map(VIWorker(m_params), zip(corpus, word_counts))
        params = map(VIWorker(m_params), zip(corpus, word_counts))
        
        old_l_bound = sum([likelihood_bound(p, m_params, d, c, sum(c)) for (p, d, c) in zip(params, corpus, word_counts)])
        print "Old bound: " + str(old_l_bound)
        
        mu = np.sum([p.lambd for p in params], axis=0) / len(corpus)
        sigma = np.sum([np.diag(p.nu_sq) + np.outer(p.lambd, p.lambd) for p in params], axis=0)
        sigma /= len(corpus)
        sigma -= np.outer(mu, mu)
        
        sigma += np.eye(len(sigma)) * 0.01 #Ridge on the principal diagonal to avoid singular matrices
        
        beta = np.zeros(m_params.beta.shape)
        for d in xrange(len(corpus)):
            for w in xrange(len(corpus[d])):
                word = corpus[d][w]
                count = word_counts[d][w]
                for i in xrange(len(beta)):
                    beta[i, word] += count * params[d].phi[w, i]
        
        for i in xrange(len(beta)):
            beta[i] /= np.sum(beta[i])
        
        m_params = Model(mu, sigma, beta)
        print m_params
        
        new_l_bound = sum([likelihood_bound(p, m_params, d, c, sum(c)) for (p, d, c) in zip(params, corpus, word_counts)])
        print "New bound: " + str(new_l_bound)
        iteration += 1

        delta = abs((new_l_bound - old_l_bound)/old_l_bound)

        yield (m_params, new_l_bound)
        
        if (delta < 1e-5 or iteration >= max_iterations):
            break
    
    #pool.close()
