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
    def __str__(self):
        return "zeta: " + str(self.zeta) + "; phi: " + str(self.phi) + \
            "lambda: " + str(self.lambd) + "; nu_sq: " + str(self.nu_sq)

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
def f_dlambda(lambd, v_params, m_params, doc, counts):
    N = sum(counts)
    
    return -(-np.linalg.inv(m_params.sigma).dot(lambd - m_params.mu) +\
        np.sum([c * v_params.phi[n] for (n, c) in zip(xrange(len(doc)), counts)]) -\
        (N/v_params.zeta) * np.exp(lambd + 0.5 * v_params.nu_sq))

#the objective function used to optimize the likelihood bound with respect to lambda
def f_lambda(lambd, v_params, m_params, doc, counts):
    v_params.lambd = lambd
    return -likelihood_bound(v_params, m_params, doc, counts)

def f_nu_sq(nu_sq, v_params, m_params, doc, counts):
    v_params.nu_sq = nu_sq
    return -likelihood_bound(v_params, m_params, doc, counts)

def f_dnu_sq(nu_sq, v_params, m_params, doc, counts):
    N = sum(counts)    
    
    inv_sigma = np.linalg.inv(m_params.sigma)
    result = 0.5 * np.diagonal(inv_sigma) + 0.5 * N / v_params.zeta * np.exp(v_params.lambd + 0.5 * nu_sq) - 0.5 / nu_sq
    return result
    
def likelihood_bound(v_params, m_params, doc, counts):
    inv_sigma = np.linalg.inv(m_params.sigma)
    
    N = sum(counts)
    
    #E_q(logp(eta|mu,sigma))
    result = 0.5 * np.log(np.linalg.det(inv_sigma))
    result -= 0.5 * (np.diag(v_params.nu_sq).dot(inv_sigma)).trace()
    lambda_mu = v_params.lambd - m_params.mu
    result -= 0.5 * lambda_mu.dot(inv_sigma.dot(lambda_mu))
    
    #E_q(logp(z|eta))
    result += sum([c * v_params.lambd[i] * v_params.phi[n, i] for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    result -= N * (1.0 / v_params.zeta * np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq)) - 1 + np.log(v_params.zeta))
    
    #E_q(logp(w|mu,z,beta))
    result += sum([c * v_params.phi[n, i] + np.log(m_params.beta[i, doc[n]]) for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    
    #H(q)
    result += np.sum(0.5 * (1 + np.log(v_params.nu_sq * 2 * np.pi)))
    result -= np.sum([c * v_params.phi[n, i] * np.log(v_params.phi[n, i]) for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    
    return result
    

"""Performs variational inference of the variational parameters on
one document given the current model parameters. Returns a VariationalParams object"""
def variational_inference(doc, counts, m_params):
    
    v_params = VariationalParams(
        zeta=10.0,\
        phi=np.zeros((len(doc), len(m_params.beta))) + 1.0/len(m_params.beta),\
        lambd=np.zeros(len(m_params.beta)),\
        nu_sq=np.ones(len(m_params.beta)))
    
    old_l_bound = likelihood_bound(v_params, m_params, doc, counts)

    while True:
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))
        
        #Maximize wrt lambda
        v_params.lambd = scipy.optimize.fmin_cg(f_lambda, v_params.lambd, f_dlambda, args=(v_params, m_params, doc, counts))
        #opt_result = scipy.optimize.minimize(f_lambda, v_params.lambd, method='BFGS', jac=f_dlambda, args=(v_params, m_params, doc, counts))        
        #v_params.lambd = opt_result.x 
        #print "max labdma: " + str(v_params.lambd)

        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))                                        
        
        #Maximize wrt nu
        nu_opt_result = scipy.optimize.fmin_l_bfgs_b(f_nu_sq, v_params.nu_sq, f_dnu_sq, args=(v_params, m_params, doc, counts), bounds = [(0, None) for _ in m_params.beta])
        v_params.nu_sq = nu_opt_result[0]
        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))
        
        #Maximize wrt phi
        for (n, c) in zip(xrange(len(doc)), counts):
            sum_n = 0
            for i in xrange(len(m_params.beta)):
                v_params.phi[n, i] = np.exp(v_params.lambd[i]) * m_params.beta[i, doc[n]]
                sum_n += c * v_params.phi[n, i]
            for i in xrange(len(m_params.beta)):
                v_params.phi[n, i] /= sum_n
        
        new_l_bound = likelihood_bound(v_params, m_params, doc, counts)

        delta = abs((new_l_bound - old_l_bound) / old_l_bound)
        
        #print old_l_bound, new_l_bound, delta
        
        old_l_bound = new_l_bound
        if (delta < 0.01):
            print "STOPPING, bound " + str(new_l_bound)
            break
        
        #print v_params.zeta, v_params.phi, v_params.lambd, v_params.nu_sq
    return v_params
        
"""Populates self.mu, self.sigma and self.beta, the latent variables of the model"""
def inference(corpus, word_counts, no_pathways, pathway_priors):
    #TODO loop until changes < threshold
    m_params = Model(np.zeros(no_pathways), np.identity(no_pathways), np.array(pathway_priors))
    
    iteration = 0
    
    while True:
        print "Iteration: " + str(iteration)
        
        params = [variational_inference(doc, count, m_params) for doc, count in zip(corpus, word_counts)]
        
        old_l_bound = sum([likelihood_bound(p, m_params, d, c) for (p, d, c) in zip(params, corpus, word_counts)])
        print "Old bound: " + str(old_l_bound)
        
        mu = np.sum([p.lambd for p in params], axis=0) / len(corpus)
        sigma = np.sum([np.diag(p.nu_sq) + np.outer(p.lambd, p.lambd) for p in params], axis=0) + np.outer(mu, mu)
        sigma /= len(corpus)
        
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
        
        new_l_bound = sum([likelihood_bound(p, m_params, d, c) for (p, d, c) in zip(params, corpus, word_counts)])
        print "New bound: " + str(new_l_bound)
        iteration += 1

        delta = abs((new_l_bound - old_l_bound)/old_l_bound)

        yield m_params
        
        if (delta < 0.001):
            break
        

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

[result.append(gendoc()) for _ in xrange(10)]

from collections import Counter
doc_words = [list(Counter(d).iterkeys()) for d in result]
doc_counts = [list(Counter(d).itervalues()) for d in result]

#priors = np.ones((K, 10))
priors = [np.random.uniform(0, 1, wlen) for _ in xrange(K)]
priors = np.array([p / sum(p) for p in priors])
ps = list(inference(doc_words, doc_counts, K, priors))


print '\n'.join([str(p) for p in ps])