# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import warnings
warnings.filterwarnings('ignore', 'Desired error not necessarily achieved due to precision loss')

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
    def __str__(self):
        return "mu: " + str(self.mu) + "; sigma: " + str(self.sigma) + "; beta: " + str(self.beta)

#the derivative of the likelihood bound with respect to lambda,
#passed to the conjugate gradient algorithm. Uses the first argument as lambda
#(ignores the lambda in v_params since the scipy optimizer manipulates the first argument)
def f_dlambda(lambd, v_params, m_params, doc):
    return -np.linalg.inv(m_params.sigma).dot(m_params.mu - lambd) -np.sum(v_params.phi, axis=0) -\
        (len(doc)/v_params.zeta) * np.exp(lambd + 0.5 * v_params.nu_sq)

#the objective function used to optimize the likelihood bound with respect to lambda
def f_lambda(lambd, v_params, m_params, doc):
    inv_sigma = np.linalg.inv(m_params.sigma)
    
    lambda_mu = lambd - m_params.mu
    result = lambd.dot(np.sum(v_params.phi, axis=0))
    result -= 0.5 * lambda_mu.dot(inv_sigma.dot(lambda_mu))
    result -= len(doc) * (1.0 / v_params.zeta * np.sum(np.exp(lambd + 0.5 * v_params.nu_sq)) - 1.0 + np.log(v_params.zeta))
    
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
    
    result += np.sum(v_params.phi.dot(v_params.lambd)) + len(doc) * (-1.0 / v_params.zeta * np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq)) + 1 - np.log(v_params.zeta))

    for n in xrange(len(doc)):
        for i in xrange(len(m_params.beta)):
            result += v_params.phi[n, i] * np.log(m_params.beta[i, doc[n]])
    
    result += 0.5 * np.sum(np.log(v_params.nu_sq) + np.log(2*np.pi) + 1)
    result -= np.sum(v_params.phi*np.log(v_params.phi))
    
    return result
    

"""Performs variational inference of the variational parameters on
one document given the current model parameters. Returns a VariationalParams object"""
def variational_inference(doc, m_params):
    
    v_params = VariationalParams(
        zeta=10.0,\
        phi=np.zeros((len(doc), len(m_params.beta))) + 1.0/len(m_params.beta),\
        lambd=np.ones(len(m_params.beta)),\
        nu_sq=np.zeros(len(m_params.beta)) + 10.0)
    
    old_l_bound = likelihood_bound(v_params, m_params, doc)

    for _ in xrange(5):
        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))
        
        #Maximize wrt phi
        for n in xrange(len(doc)):
            sum_n = 0
            for i in xrange(len(m_params.beta)):
                v_params.phi[n, i] = np.exp(v_params.lambd[i]) * m_params.beta[i, doc[n]]
                sum_n += v_params.phi[n, i]
            for i in xrange(len(m_params.beta)):
                v_params.phi[n, i] /= sum_n
        
        #Maximize wrt lambda
        v_params.lambd = scipy.optimize.fmin_cg(f_lambda, v_params.lambd, f_dlambda, args=(v_params, m_params, doc))
        #print "max labdma: " + str(v_params.lambd)
                        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))
        
        #Maximize wrt nu
        #Use Newton's method on every item in nu on the derivative to find the maximum
        for i in xrange(len(v_params.nu_sq)):
            v_params.nu_sq[i] = scipy.optimize.newton(f_dnu_sq, 0.1, f_d2nu_sq, args=(i, v_params, m_params, doc))
        #print "max nu_sq: " + str(v_params.nu_sq)
        
        new_l_bound = likelihood_bound(v_params, m_params, doc)
        #print old_l_bound, new_l_bound
        
        old_l_bound = new_l_bound
        
        #print v_params.zeta, v_params.phi, v_params.lambd, v_params.nu_sq
    return v_params
    
def model_inference_step(m_params, corpus, word_counts):
    params = [variational_inference(doc, m_params) for doc in corpus]
    
    mu = np.sum([p.lambd for p in params], axis=0) / len(corpus)
    sigma = np.sum([np.diag(p.nu_sq) + np.outer(p.lambd - m_params.mu, p.lambd - m_params.mu) for p in params], axis=0) / len(corpus)
    
    beta = np.zeros(m_params.beta.shape)
    for d in xrange(len(corpus)):
        for w in xrange(len(corpus[d])):
            word = corpus[d][w]
            count = word_counts[d][w]
            for i in xrange(len(m_params.beta)):
                phi = params[d].phi
                beta[i, word] += count * phi[w, i]
    
    for i in xrange(len(m_params.beta)):
        m_params.beta[i] /= np.sum(m_params.beta[i])
        
    #for p in params:
     #   print p.zeta, p.phi, p.lambd, p.nu_sq
    
    print "mu", m_params.mu, "sigma", m_params.sigma, "beta", m_params.beta
    
    return Model(mu, sigma, beta)
        
"""Populates self.mu, self.sigma and self.beta, the latent variables of the model"""
def inference(corpus, word_counts, no_pathways, pathway_priors):
    #TODO loop until changes < threshold
    m_params = Model(np.zeros(no_pathways), np.identity(no_pathways), np.array(pathway_priors))
    
    for _ in xrange(5):
        m_params = model_inference_step(m_params, corpus, word_counts)
        yield m_params
    

def f(eta):
    return np.exp(eta) / np.sum(np.exp(eta))

#corpus is a list of word counts
#corpus = [np.array([0,1,2]), np.array([2,3,4])]
#no_pathways = 2
#pathway_priors = np.ones((2, 5))
#word_counts = [np.array([10, 10, 10]), np.array([10, 10, 10])]

#

wlen = 10 #vocabulary length
vocabulary = xrange(wlen)

K = 5
mu = np.random.uniform(0, 1, K)
sigma = np.identity(K)

beta = [np.random.uniform(0, 1, wlen) for _ in xrange(K)]
beta = [b / sum(b) for b in beta]

N_d = 100 #document length

eta_d = np.random.multivariate_normal(mu, sigma)

def gendoc():
    
    document = []
    
    for n in xrange(N_d):
        Z_dn = beta[np.flatnonzero(np.random.multinomial(1, f(eta_d)))[0]]
        W_dn = np.random.choice(xrange(len(Z_dn)), p=Z_dn)
        document.append(W_dn)

    return document

result = [] 

[result.append(gendoc()) for _ in xrange(100)]

from collections import Counter
doc_words = [list(Counter(d).iterkeys()) for d in result]
doc_counts = [list(Counter(d).itervalues()) for d in result]

ps = list(inference(doc_words, doc_counts, K, np.ones((K, 10))))


print '\n'.join([str(p) for p in ps])