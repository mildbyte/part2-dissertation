# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

class VariationalParams():
    def __init__(self, zeta, phi, lambd, nu_sq):
        self.zeta = zeta
        self.phi = phi
        self.lambd = lambd
        self.nu_sq = nu_sq

class Model():
    def __init__(self, mu, sigma, beta):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta

#the derivative of the likelihood bound with respect to lambda,
#passed to the conjugate gradient algorithm. Uses the first argument as lambda
#(ignores the lambda in v_params since the scipy optimizer manipulates the first argument)
def f_dlambda(lambd, v_params, m_params, doc):
    return np.linalg.inv(m_params.sigma).dot(lambd - m_params.mu) - np.sum(v_params.phi, axis=0)\
        + (len(doc)/v_params.zeta) * np.exp(lambd + 0.5 * v_params.nu_sq)

#the objective function used to optimize the likelihood bound with respect to lambda
def f_lambda(lambd, v_params, m_params, doc):
    inv_sigma = np.linalg.inv(m_params.sigma)
    
    lambda_mu = lambd - m_params.mu
    result = np.sum(v_params.phi.dot(lambd))
    result -= 0.5 * lambda_mu.dot(inv_sigma.dot(lambda_mu))
    result -= len(doc) / v_params.zeta * np.sum(np.exp(lambd + 0.5 * v_params.nu_sq)) - 1.0 + np.log(v_params.zeta)
    
    return -result

def f_dnu_sq(nu_sq, i, v_params, m_params, doc):
    inv_sigma = np.linalg.inv(m_params.sigma)
    return -0.5 * inv_sigma[(i, i)] - 0.5 * len(doc) / v_params.zeta * np.exp(v_params.lambd[i] + 0.5 * nu_sq) + 0.5 / nu_sq

def f_d2nu_sq(nu_sq, i, v_params, m_params, doc):
    return -0.25 * len(doc) / v_params.zeta * np.exp(v_params.lambd[i] + 0.5 * nu_sq) - 0.5 / (nu_sq * nu_sq)

def likelihood_bound(v_params, m_params, doc):
    inv_sigma = np.linalg.inv(m_params.sigma)
    
    result = 0.5 * np.log(np.linalg.det(inv_sigma)) - 0.5 * len(m_params.beta) * np.log(2*np.pi)

    lambda_mu = v_params.lambd - m_params.mu
    
    result -= 0.5 * (np.diag(v_params.nu_sq).dot(inv_sigma)).trace()
    result -= 0.5 * lambda_mu.dot(inv_sigma.dot(lambda_mu))
    result += np.sum(v_params.phi.dot(v_params.lambd)) - 1.0 / v_params.zeta * np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq)) + 1 - np.log(v_params.zeta)

    for n in xrange(len(doc)):
        for i in xrange(len(m_params.beta)):
            result += np.exp(v_params.phi[i]) * m_params.beta[i, doc[n]]
    
    result += 0.5 * np.sum(np.log(v_params.nu_sq) + np.log(2*np.pi) + 1)
    result -= np.sum(v_params.phi*np.log(v_params.phi))
    
    return result
    

"""Performs variational inference of the variational parameters on
one document given the current model parameters. Returns a VariationalParams object"""
def variational_inference(doc, m_params):
    
    v_params = VariationalParams(
        0.0, np.zeros((len(doc), len(m_params.beta))),\
        np.ones(len(m_params.beta)), np.ones(len(m_params.beta)))

    for _ in xrange(10):
        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))
        
        #Maximize wrt phi
        for n in xrange(len(doc)):
            for i in xrange(len(m_params.beta)):
                v_params.phi[n, i] = np.exp(v_params.lambd[i]) * m_params.beta[i, doc[n]]
        
        #Maximize wrt lambda
        v_params.lambd = scipy.optimize.fmin_cg(f_lambda, v_params.lambd, f_dlambda, args=(v_params, m_params, doc))
        #print "max labdma: " + str(v_params.lambd)
                        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))        
        
        #Maximize wrt nu
        #Use Newton's method on every item in nu on the derivative to find the maximum
        for i in xrange(len(v_params.nu_sq)):
#            X = np.linspace(0, 1)
#            plt.plot(X, f_dnu_sq(X, i, v_params, m_params, doc))
            
            v_params.nu_sq[i] = scipy.optimize.newton(f_dnu_sq, 0.1, f_d2nu_sq, args=(i, v_params, m_params, doc))
        
        #print v_params.zeta, v_params.phi, v_params.lambd, v_params.nu_sq
    return v_params
    
def model_inference_step(m_params, corpus, word_counts):
    params = [variational_inference(doc, m_params) for doc in corpus]
    
    m_params.mu = np.sum([p.lambd for p in params], axis=0) / len(corpus)
    m_params.sigma = np.sum([np.diag(p.nu_sq) + np.outer(p.lambd - m_params.mu, p.lambd - m_params.mu) for p in params], axis=0) / len(corpus)
    
    m_params.beta.fill(0)
    for d in xrange(len(corpus)):
        for w in xrange(len(corpus[d])):
            word = corpus[d][w]
            count = word_counts[d][w]
            for i in xrange(len(m_params.beta)):
                m_params.beta[i, word] += count * params[d].phi[w, i]
    
    for i in xrange(len(m_params.beta)):
        m_params.beta[i] /= np.sum(m_params.beta[i])
        
    #for p in params:
     #   print p.zeta, p.phi, p.lambd, p.nu_sq
    
    print m_params.mu, m_params.sigma, m_params.beta
        
"""Populates self.mu, self.sigma and self.beta, the latent variables of the model"""
def inference(corpus, word_counts, no_pathways, pathway_priors):
    #TODO loop until changes < threshold
    m_params = Model(np.ones(no_pathways), np.identity(no_pathways), np.array(pathway_priors))
    
    for _ in xrange(50):
        model_inference_step(m_params, corpus, word_counts)
    
    return m_params
    

def f(eta):
    return np.exp(eta) / np.sum(np.exp(eta))

#corpus is a list of word counts
corpus = [np.array([0,1,2]), np.array([2,3,4])]
no_pathways = 2
pathway_priors = np.ones((2, 5))
word_counts = [np.array([10, 10, 10]), np.array([10, 10, 10])]

p = inference(corpus, word_counts, no_pathways, pathway_priors)

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

