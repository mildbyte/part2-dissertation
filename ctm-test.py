# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats
import scipy.misc
from collections import Counter


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
        
        self.inv_sigma = np.linalg.inv(sigma)
    def __str__(self):
        return "mu: " + str(self.mu) + "; sigma: " + str(self.sigma) + "; beta: " + str(self.beta)

#the derivative of the likelihood bound with respect to lambda,
#passed to the conjugate gradient algorithm. Uses the first argument as lambda
#(ignores the lambda in v_params since the scipy optimizer manipulates the first argument)
def f_dlambda(lambd, v_params, m_params, doc, counts):
    N = sum(counts)
    
    return -(-m_params.inv_sigma.dot(lambd - m_params.mu) +\
        np.sum([c * v_params.phi[n] for (n, c) in zip(xrange(len(doc)), counts)], axis=0) -\
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
    
    result = 0.5 * np.diagonal(m_params.inv_sigma) + 0.5 * N / v_params.zeta * np.exp(v_params.lambd + 0.5 * nu_sq) - 0.5 / nu_sq
    return result
    
def likelihood_bound(v_params, m_params, doc, counts):
    N = sum(counts)
    
    #E_q(logp(eta|mu,sigma))
    result = 0.5 * np.log(np.linalg.det(m_params.inv_sigma))
    result -= 0.5 * np.log(2 * np.pi) * len(m_params.beta)
    result -= 0.5 * (np.diag(v_params.nu_sq).dot(m_params.inv_sigma)).trace()
    lambda_mu = v_params.lambd - m_params.mu
    result -= 0.5 * lambda_mu.dot(m_params.inv_sigma.dot(lambda_mu))
    
    #E_q(logp(z|eta))
    result += sum([c * v_params.lambd[i] * v_params.phi[n, i] for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    result -= N * (1.0 / v_params.zeta * np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq)) - 1 + np.log(v_params.zeta))
    
    #E_q(logp(w|mu,z,beta))
    result += sum([c * v_params.phi[n, i] * np.log(m_params.beta[i, doc[n]]) for (n, c) in zip(xrange(len(doc)), counts) for i in xrange(len(m_params.beta))])
    
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
        lambd=np.ones(len(m_params.beta)),\
        nu_sq=np.ones(len(m_params.beta)))
    
    old_l_bound = likelihood_bound(v_params, m_params, doc, counts)

    while True:
        #Maximize wrt zeta
        v_params.zeta = np.sum(np.exp(v_params.lambd + 0.5 * v_params.nu_sq))
        
        #Maximize wrt lambda
   #     v_params.lambd = scipy.optimize.fmin_cg(f_lambda, v_params.lambd, f_dlambda, args=(v_params, m_params, doc, counts))
        opt_result = scipy.optimize.minimize(f_lambda, v_params.lambd, method='BFGS', jac=f_dlambda, args=(v_params, m_params, doc, counts))        
        v_params.lambd = opt_result.x 
        
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
                sum_n += v_params.phi[n, i]
            for i in xrange(len(m_params.beta)):
                v_params.phi[n, i] /= sum_n
        
        new_l_bound = likelihood_bound(v_params, m_params, doc, counts)

        delta = abs((new_l_bound - old_l_bound) / old_l_bound)
        
        old_l_bound = new_l_bound
        if (delta < 1e-5):
            break
        
    return v_params
        
"""Populates self.mu, self.sigma and self.beta, the latent variables of the model"""
def inference(corpus, word_counts, no_pathways, pathway_priors):
    m_params = Model(np.zeros(no_pathways), np.identity(no_pathways), np.array(pathway_priors))
    
    iteration = 0
    
    while True:
        print "Iteration: " + str(iteration)
        
        params = [variational_inference(doc, count, m_params) for doc, count in zip(corpus, word_counts)]
        
        old_l_bound = sum([likelihood_bound(p, m_params, d, c) for (p, d, c) in zip(params, corpus, word_counts)])
        print "Old bound: " + str(old_l_bound)
        
        mu = np.sum([p.lambd for p in params], axis=0) / len(corpus)
        sigma = np.sum([np.diag(p.nu_sq) + np.outer(p.lambd - mu, p.lambd - mu) for p in params], axis=0) / len(corpus)
        
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

        yield (m_params, new_l_bound)
        
        if (delta < 1e-5):
            break

"""Sampling of the likelihood based on the variational posterior"""
def sample_term(v_params, m_params, doc, counts, eta):
    t1 = 0.5 * np.log(np.linalg.det(m_params.inv_sigma))
    t1 -= 0.5 * (np.diag(v_params.nu_sq).dot(m_params.inv_sigma)).trace()
    lambda_mu = v_params.lambd - m_params.mu
    t1 -= 0.5 * lambda_mu.dot(m_params.inv_sigma.dot(lambda_mu))
    
    theta = np.exp(eta)
    theta /= sum(theta)
    
    for (n, c) in zip(xrange(len(doc)), counts):
        t1 += c * np.log(np.sum(theta * m_params.beta[:, doc[n]]))
        
    t2 = np.sum(np.log(scipy.stats.multivariate_normal.pdf(eta - v_params.lambd, np.zeros(len(eta)), np.diag(np.sqrt(v_params.nu_sq)))))
    
    return t1 - t2

def sample_lhood(v_params, m_params, doc, counts):
    nsamples = 10000
    
    def terms():
        for _ in xrange(nsamples):
            eta = np.random.multivariate_normal(v_params.lambd, np.diag(np.sqrt(v_params.nu_sq)))
            yield sample_term(v_params, m_params, doc, counts, eta)
            
    return scipy.misc.logsumexp(np.array(terms())) - np.log(nsamples)
    
def expected_theta(v_params, m_params, doc, counts):
    nsamples = 100
    
    def terms():
            
        for _ in xrange(nsamples):
            eta = np.random.multivariate_normal(v_params.lambd, np.diag(np.sqrt(v_params.nu_sq)))
            
            theta = np.exp(eta)
            theta /= sum(theta)
        
            w = sample_term(v_params, m_params, doc, counts, eta)
            
            yield w + np.log(theta)
    
    samples = list(terms())

    e_theta = scipy.misc.logsumexp(samples, axis=0) - np.log(nsamples)
    norm = scipy.misc.logsumexp(e_theta)
    return np.exp(e_theta - norm)
        
def f(eta):
    return np.exp(eta) / np.sum(np.exp(eta))
    
def generate_random_corpus(voc_len, K, N_d, no_docs):
    def gendoc():
        
        eta_d = np.random.multivariate_normal(mu, sigma)
        
        document = []    
        for n in xrange(N_d):
            
            Z_dn = beta[np.flatnonzero(np.random.multinomial(1, f(eta_d)))[0]]
            W_dn = np.random.choice(xrange(len(Z_dn)), p=Z_dn)
            document.append(W_dn)
    
        return document, eta_d
        
    mu = np.random.uniform(0, 1, K)
    sigma = np.identity(K)
    
    beta = [np.random.uniform(0, 1, voc_len) for _ in xrange(K)]
    beta = [b / sum(b) for b in beta]
    
    doc_words = []
    doc_counts = []
    doc_thetas = []

    for _ in xrange(no_docs):
        doc, eta_d = gendoc()
        c = Counter(doc)
        
        doc_words.append(list(c.iterkeys()))
        doc_counts.append(list(c.itervalues()))
        doc_thetas.append(f(eta_d))

    return doc_words, doc_counts, doc_thetas, mu, sigma, beta

def error_measure(m_params, doc_words, doc_counts, doc_thetas):
    thetas = np.array([expected_theta(variational_inference(d, c, m_params), m_params, d, c) for (d, c) in zip(doc_words, doc_counts)])
    return np.mean(np.linalg.norm(thetas - doc_thetas, axis=1))
    
def validation():
    corpus = zip(doc_words, doc_counts, doc_thetas)
    np.random.shuffle(corpus)
    
    split = int(len(corpus)*0.8)
    train_data = corpus[:split]
    test_data = corpus[split:]
    
    (train_words, train_counts, train_thetas) = zip(*train_data)

    priors = [np.random.uniform(0, 1, voc_len) for _ in xrange(K)]
    priors = np.array([p / sum(p) for p in priors])
    
    ps = list(inference(train_words, train_counts, K, priors))
    m_params = ps[-1][0]
    
    (test_words, test_counts, test_thetas) = zip(*test_data)
    

voc_len = 10
K = 2
N_d = 100
no_docs = 10

doc_words, doc_counts, doc_thetas, mu, sigma, beta = generate_random_corpus(voc_len, K, N_d, no_docs)

priors = [np.random.uniform(0, 1, voc_len) for _ in xrange(K)]
priors = np.array([p / sum(p) for p in priors])

ps = list(inference(doc_words, doc_counts, K, priors))
np.set_printoptions(precision=2)

m_params = ps[-1][0]

thetas = np.array([expected_theta(variational_inference(d, c, m_params), m_params, d, c) for (d, c) in zip(doc_words, doc_counts)])

theta_differences = thetas - doc_thetas

theta_diff_sizes = np.linalg.norm(theta_differences, axis=1)

baseline = np.random.multivariate_normal(mu, sigma, size=no_docs)
baseline = np.exp(baseline) / np.sum(np.exp(baseline), axis=1)[:, None]
baseline_diff = baseline - doc_thetas
baseline_diff_sizes = np.linalg.norm(baseline_diff, axis=1)


def plot_cdf(arr):
    counts, edges = np.histogram(arr, normed=True, bins=1000)
    cdf = np.cumsum(counts)
    cdf /= max(cdf)
    plt.plot(edges[1:], cdf)

plot_cdf(theta_diff_sizes)
plot_cdf(baseline_diff_sizes)

plt.legend([r"Inferred $\theta$", r"Baseline $\theta$ (random)"])
plt.xlabel("Norm of $\\theta_{inf}$ - $\\theta_{ref}$")
plt.ylabel("Proportion of errors below a given norm (the CDF)")