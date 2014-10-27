# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

import numpy as np
import matplotlib.pyplot as plt

class VariationalParams():
    def __init__(self, zeta, phi, lambd, nu):
        self.zeta = zeta
        self.phi = phi
        self.lambd = lambd
        self.nu = nu

class CTM():
    """corpus: a matrix such that corpus_(n, m) is the frequency of the mth word in the nth document"""
    def __init__(self, corpus, no_pathways, pathway_priors):
        self.corpus = corpus
        self.no_pathways = no_pathways
        self.pathway_priors = pathway_priors
    
    def 
    def fdlambda(d, mu, sigma, zeta, phi, lambd, nu):
       return -np.invert(sigma)*(lambd - mu) + np.sum(phi, axis=0) - (len(self.corpus[d])/zeta) * np.exp(lambd + nu^2/2)

    """Performs variational inference of the variational parameters on
    one document given the current model parameters. Returns a VariationalParams object"""
    def variational_inference(self, d, mu, sigma, beta):
        
        zeta = 0.0
        phi = np.zeros((len(self.corpus[d]), self.no_pathways))
        lambd = np.zeros(self.no_pathways)
        nu = np.zeros(self.no_pathways)
        

        zeta = np.sum(np.exp(lambd + nu^2/2))
        
        for n in xrange(len(self.corpus[d])):
            for i in xrange(self.no_pathways):
                phi[n, i] = np.exp(lambd[i])*beta[i, self.corpus[d, n]]
        
                
        
        return (zeta, phi, lambd, nu)
        
        

    """Populates self.mu, self.sigma and self.beta, the latent variables of the model"""
    def inference(self):
        pass
    
    

def f(eta):
    return np.exp(eta) / np.sum(np.exp(eta))

wlen = 10
    
vocabulary = xrange(wlen)

K = 5
mu = np.random.uniform(0, 1, K)
sigma = np.identity(K)

beta = [np.random.uniform(0, 1, wlen) for _ in xrange(K)]
beta = [b / sum(b) for b in beta]

N_d = 1000

eta_d = np.random.multivariate_normal(mu, sigma)
eta_d[xrange(1,5)] = -100000000

def gendoc():
    
    document = []
    
    for n in xrange(N_d):
        Z_dn = beta[np.flatnonzero(np.random.multinomial(1, f(eta_d)))[0]]
        W_dn = np.random.choice(xrange(len(Z_dn)), p=Z_dn)
        document.append(W_dn)

    return document

result = [] 

#[result.extend(gendoc()) for _ in xrange(100)]

plt.hist(result)

res2 = np.array(result)
res2 = np.histogram(res2, normed=True)