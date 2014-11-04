# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

class VariationalParams():
    def __init__(self, zeta, phi, lambd, nu):
        self.zeta = zeta
        self.phi = phi
        self.lambd = lambd
        self.nu = nu

class Model():
    def __init__(self, mu, sigma, beta):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta

#the derivative of the likelihood bound with respect to lambda,
#passed to the conjugate gradient algorithm. Uses the first argument as lambda
#(ignores the lambda in v_params since the scipy optimizer manipulates the first argument)
def f_dlambda(lambd, v_params, m_params, doc):
    return -np.linalg.inv(m_params.sigma)*(lambd[0] - m_params.mu) + np.sum(v_params.phi, axis=0)\
        - (len(doc)/v_params.zeta) * np.exp(lambd[0] + 0.5 * v_params.nu**2)

#the objective function used to optimize the likelihood bound with respect to lambda
def f_lambda(lambd, v_params, m_params, doc):
    v_params.lambd = lambd
    return -likelihood_bound(v_params, m_params, doc)

def likelihood_bound(v_params, m_params, doc):
    inv_sigma = np.linalg.inv(m_params.sigma)
    
    result = 0.5 * np.log(np.linalg.det(inv_sigma)) - 0.5 * len(m_params.beta) * np.log(2*np.pi)

    lambda_mu = v_params.lambd - m_params.mu
    
    result -= 0.5 * (np.diag(v_params.nu ** 2).dot(inv_sigma)).trace()
    result -= 0.5 * lambda_mu.dot(inv_sigma.dot(lambda_mu))
    result += np.sum(v_params.phi.dot(v_params.lambd)) - 1.0 / v_params.zeta * np.sum(np.exp(v_params.lambd + v_params.nu**2)) + 1 - np.log(v_params.zeta)

    for n in xrange(len(doc)):
        for i in xrange(len(m_params.beta)):
            result += np.exp(v_params.lambd[i]) * m_params.beta[i, doc[n]]
    
    result += 0.5 * np.sum(np.log(v_params.nu**2) + np.log(2*np.pi) + 1)
    result -= np.sum(v_params.phi*np.log(v_params.phi))
    
    return result
    

"""Performs variational inference of the variational parameters on
one document given the current model parameters. Returns a VariationalParams object"""
def variational_inference(doc, m_params):
    
    v_params = VariationalParams(
        0.0, np.zeros((len(doc), len(m_params.beta))),\
        np.zeros(len(m_params.beta)), np.zeros(len(m_params.beta)))
    
    #Maximize wrt zeta
    v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu**2))
    
    #Maximize wrt phi
    for n in xrange(len(doc)):
        for i in xrange(len(m_params.beta)):
            v_params.phi[n, i] = np.exp(v_params.lambd[i]) * m_params.beta[i, doc[n]]
    
    #Maximize wrt lambda
    v_params.lambd = scipy.optimize.fmin_cg(f_lambda, v_params.lambd, f_dlambda, args=(v_params, m_params, doc))
    
    #TODO newton method on every coordinate for the derivative to find its zero (and the function max)
    
    return v_params
    
def model_inference_step(m_params, corpus):
    params = [variational_inference(doc, m_params) for doc in corpus]
    
    m_params.mu = sum([p.lambd for p in params]) / len(corpus)
    #TODO calculate sigma and beta
        
"""Populates self.mu, self.sigma and self.beta, the latent variables of the model"""
def inference(corpus, no_pathways, pathway_priors):
    #TODO loop until changes < threshold
    m_params = Model(np.zeros(no_pathways), np.identity(no_pathways), np.array(pathway_priors))
    model_inference_step(m_params, corpus)
    

def f(eta):
    return np.exp(eta) / np.sum(np.exp(eta))

#corpus is a list of word counts
corpus = [np.array([0,1]), np.array([0,1])]
no_pathways = 2
pathway_priors = np.ones((2, 2))

inference(corpus, no_pathways, pathway_priors)

#wlen = 10
#    
#vocabulary = xrange(wlen)
#
#K = 5
#mu = np.random.uniform(0, 1, K)
#sigma = np.identity(K)
#
#beta = [np.random.uniform(0, 1, wlen) for _ in xrange(K)]
#beta = [b / sum(b) for b in beta]
#
#N_d = 1000
#
#eta_d = np.random.multivariate_normal(mu, sigma)
#eta_d[xrange(1,5)] = -100000000
#
#def gendoc():
#    
#    document = []
#    
#    for n in xrange(N_d):
#        Z_dn = beta[np.flatnonzero(np.random.multinomial(1, f(eta_d)))[0]]
#        W_dn = np.random.choice(xrange(len(Z_dn)), p=Z_dn)
#        document.append(W_dn)
#
#    return document
#
#result = [] 
#
##[result.extend(gendoc()) for _ in xrange(100)]
#
#plt.hist(result)
#
#res2 = np.array(result)
#res2 = np.histogram(res2, normed=True)

