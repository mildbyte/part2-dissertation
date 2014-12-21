# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 17:47:36 2014

@author: mildbyte
"""

import numpy as np
import scipy.optimize
import numexpr as ne

log2pi = np.log(2 * np.pi)


class VariationalParams():
    def __init__(self, zeta, phi, lambd, nu_sq, doc_counts):
        self.zeta = zeta
        self.phi = phi
        self.lambd = lambd
        self.nu_sq = nu_sq
        
        self.weighted_sum_phi = doc_counts.dot(phi)
    def update_ws_phi(self, doc_counts):
        self.weighted_sum_phi = doc_counts.dot(self.phi) #Must be called every time phi is changed
    def __str__(self):
        return "zeta: " + str(self.zeta) + "; phi: " + str(self.phi) + \
            "lambda: " + str(self.lambd) + "; nu_sq: " + str(self.nu_sq)

"""First derivative of f_lambda with respect to lambda.
   Uses the first parameter (passed by the optimizer) as lambda."""
def f_dlambda(lambd, v_params, m_params, doc, counts, N):
    return -(-m_params.inv_sigma.dot(lambd - m_params.mu) +\
        #np.sum([c * v_params.phi[n] for (n, c) in zip(xrange(len(doc)), counts)], axis=0) -\
        v_params.weighted_sum_phi -\
        N * np.exp(lambd + 0.5 * v_params.nu_sq - np.log(v_params.zeta)))
        #shift zeta inside to make exp overflow happen less often

"""The objective function used to optimize the likelihood bound with respect to lambda.
   Same as the negated likelihood bound with only lambda-dependent terms."""
def f_lambda(lambd, v_params, m_params, doc, counts, N):
    #E_q(logp(eta|mu,sigma))
    lambda_mu = lambd - m_params.mu
    result = 0.5 * lambda_mu.dot(m_params.inv_sigma.dot(lambda_mu))
    
    #E_q(logp(z|eta))
    #result -= sum([c * lambd[i] * v_params.phi[n, i] for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    result -= lambd.dot(v_params.weighted_sum_phi)
    result += N * np.sum(np.exp(lambd + 0.5 * v_params.nu_sq - np.log(v_params.zeta)))
    #shift zeta inside to make exp overflow happen less often
    
    return result

"""Objective function used to optimize the bound with respect to nu_sq.
   Same as the negated likelihood bound with only the nu_sq terms."""
def f_nu_sq(nu_sq, v_params, m_params, doc, counts, N):
    #E_q(logp(eta|mu,sigma))
    result = 0.5 * (np.diag(nu_sq).dot(m_params.inv_sigma)).trace()
    
    #E_q(logp(z|eta))
    result += N * np.sum(np.exp(v_params.lambd + 0.5 * nu_sq - np.log(v_params.zeta)))
    
    #H(q)
    result -= np.sum(0.5 * (1 + np.log(nu_sq * 2 * np.pi)))
    
    return result    
    
"""The first derivative of f_nu_sq"""
def f_dnu_sq(nu_sq, v_params, m_params, doc, counts, N):
    result = 0.5 * np.diagonal(m_params.inv_sigma) +\
        0.5 * N * np.exp(v_params.lambd + 0.5 * nu_sq - np.log(v_params.zeta)) - 0.5 / nu_sq
    return result
    
def likelihood_bound(v_params, m_params, doc, counts, N):
    #E_q(logp(eta|mu,sigma))
    result = 0.5 * np.linalg.slogdet(m_params.inv_sigma)[1] #logdet avoids overflow (as opposed to log(det(inv_sigma)))
    result -= 0.5 * log2pi * len(m_params.beta)
    result -= 0.5 * (np.diag(v_params.nu_sq).dot(m_params.inv_sigma)).trace()
    lambd = v_params.lambd
    mu = m_params.mu
    lambda_mu = ne.evaluate("lambd - mu")
    result -= 0.5 * lambda_mu.dot(m_params.inv_sigma.dot(lambda_mu))
    
    #E_q(logp(z|eta))
    result += v_params.weighted_sum_phi.dot(v_params.lambd)
    nu_sq = v_params.nu_sq
    zeta = v_params.zeta
    result -= N * (ne.evaluate("sum(exp(lambd + 0.5 * nu_sq - log(zeta)))") - 1 + np.log(v_params.zeta))
    
    #E_q(logp(w|mu,z,beta))
    phi = v_params.phi
    betaTdoc = m_params.beta.T[doc]
    result += np.sum(np.dot(counts, ne.evaluate("phi * log(betaTdoc)")))
    
    #H(q)
    result += ne.evaluate("sum(0.5 * (1 + log2pi + log(nu_sq)))")
    result -= np.sum(np.dot(counts, ne.evaluate("phi * log(phi)")))
    
    return result
    

"""Performs variational inference of the variational parameters on
one document given the current model parameters. Returns a VariationalParams object"""
def variational_inference(doc, counts, m_params, max_iterations=100, initial_v_params=None):
    if initial_v_params:
        v_params = initial_v_params
    else:
        v_params = VariationalParams(
            zeta=10.0,\
            phi=np.zeros((len(doc), len(m_params.beta))) + 1.0/len(m_params.beta),\
            lambd=np.ones(len(m_params.beta)),\
            nu_sq=np.ones(len(m_params.beta)),\
            doc_counts=counts)
            
    N = np.sum(counts)
    bounds = [(0.001, None)] * len(m_params.beta)
    
    old_l_bound = likelihood_bound(v_params, m_params, doc, counts, N)
    iteration = 0

    while True:
        iteration += 1        
        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))
        
        #Maximize wrt lambda
   #     v_params.lambd = scipy.optimize.fmin_cg(f_lambda, v_params.lambd, f_dlambda, args=(v_params, m_params, doc, counts))
        opt_result = scipy.optimize.minimize(f_lambda, v_params.lambd, method='CG', jac=f_dlambda, args=(v_params, m_params, doc, counts, N))
        v_params.lambd = opt_result.x 
        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))                                        
        
        #Maximize wrt nu
        nu_opt_result = scipy.optimize.minimize(f_nu_sq, v_params.nu_sq, method='L-BFGS-B', jac=f_dnu_sq, args=(v_params, m_params, doc, counts, N), bounds = bounds)
        v_params.nu_sq = nu_opt_result.x
        
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))
        
        #Maximize wrt phi
        v_params.phi = np.multiply(np.exp(v_params.lambd), m_params.beta.T[doc])
        phi_norm = np.sum(v_params.phi, axis=1)
        v_params.phi /= phi_norm[:, np.newaxis]        
        v_params.update_ws_phi(counts)
        
        new_l_bound = likelihood_bound(v_params, m_params, doc, counts, N)
        delta = abs((new_l_bound - old_l_bound) / old_l_bound)
        old_l_bound = new_l_bound
        if (delta < 1e-5 or iteration >= max_iterations):
            break
        
    return v_params
