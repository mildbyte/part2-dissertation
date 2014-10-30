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

#the derivative of the likelihood bound with respect to lambda,
#passed to the conjugate gradient algorithm. Uses the first argument as lambda
#(ignores the lambda in v_params since the scipy optimizer manipulates the first argument)
def f_dlambda(lambd, v_params, self, d, mu, sigma, beta):
    return -np.invert(sigma)*(lambd[0] - mu) + np.sum(v_params.phi, axis=0) - (len(self.corpus[d])/v_params.zeta) * np.exp(lambd[0] + v_params.nu^2/2)

#the objective function used to optimize the likelihood bound with respect to lambda
def f_lambda(lambd, v_params, self, d, mu, sigma, beta):
    v_params.lambd = lambd
    return -self.likelihood_bound(d, v_params, mu, sigma, beta)

class CTM():
    """corpus: a matrix such that corpus_(n, m) is the frequency of the mth word in the nth document"""
    def __init__(self, corpus, no_pathways, pathway_priors):
        self.corpus = corpus
        self.no_pathways = no_pathways
        self.pathway_priors = pathway_priors
    
    def likelihood_bound(self, d, v_params, mu, sigma, beta):
        inv_sigma = np.invert(sigma)
        
        result = 0.5 * np.log(np.linalg.det(inv_sigma)) - self.no_pathways / 2 * np.log(2*np.pi)
    
        lambda_mu = v_params.lambd - mu
        
        result -= (np.diag(v_params.nu ^ 2) * inv_sigma).transpose + lambda_mu * inv_sigma * lambda_mu.transpose
        result += v_params.phi * v_params.lambd - 1.0 / v_params.zeta * np.sum(np.exp(v_params.lambd + v_params.nu^2)) + 1 - np.log(v_params.zeta)

        for n in xrange(len(self.corpus[d])):
            for i in xrange(self.no_pathways):
                result += np.exp(v_params.lambd[i]) * beta[i, self.corpus[d, n]]
        
        result += 0.5 * np.sum(np.log(v_params.nu^2) + np.log(2*np.pi) + 1)
        result -= np.sum(v_params.phi*np.log(v_params.phi))
        
        return result

    """Performs variational inference of the variational parameters on
    one document given the current model parameters. Returns a VariationalParams object"""
    def variational_inference(self, d, mu, sigma, beta):
        
        v_params = VariationalParams(
            0.0, np.zeros((len(self.corpus[d]), self.no_pathways)),\
            np.zeros(self.no_pathways), np.zeros(self.no_pathways))
        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + v_params.nu^2/2))
        
        #Maximize wrt phi
        for n in xrange(len(self.corpus[d])):
            for i in xrange(self.no_pathways):
                v_params.phi[n, i] = np.exp(v_params.lambd[i]) * beta[i, self.corpus[d, n]]
        
        #Maximize wrt lambda
        v_params.lambd = scipy.optimize.fmin_cg(f_lambda, v_params.lambd, f_dlambda, (v_params, self, d, mu, sigma, beta))
        
        #TODO newton method on every coordinate for the derivative to find its zero (and the function max)
        
        return v_params
        
        

    """Populates self.mu, self.sigma and self.beta, the latent variables of the model"""
    def inference(self):
        params = [self.variational_inference(d, self.mu, self.sigma, self.beta) for d in xrange(len(self.corpus))]
        
        self.mu = sum([p.lambd for p in params]) / len(self.corpus)
    
    

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