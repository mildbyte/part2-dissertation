# -*- coding: utf-8 -*-
"""
Created on Sun Dec  7 17:56:02 2014

@author: mildbyte
"""

import numpy as np
import scipy.stats
from variational_inference import variational_inference
from expectation_maximization import expectation_maximization
from inference import expected_theta
from collections import Counter
     
def f(eta):
    return np.exp(eta) / np.sum(np.exp(eta))
    
def generate_random_corpus(voc_len, K, N_d, no_docs):
    def gendoc(mu, sigma, beta):
        
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
    for i in xrange(K):
        beta[i][i] = 0
    beta = [b / sum(b) for b in beta]
    
    doc_words = []
    doc_counts = []
    doc_thetas = []

    for _ in xrange(no_docs):
        doc, eta_d = gendoc(mu, sigma, beta)
        c = Counter(doc)
        
        doc_words.append(list(c.iterkeys()))
        doc_counts.append(list(c.itervalues()))
        doc_thetas.append(f(eta_d))

    return doc_words, doc_counts, doc_thetas, mu, sigma, beta

def error_measure(m_params, doc_words, doc_counts, doc_thetas):
    thetas = np.array([expected_theta(variational_inference(d, c, m_params), m_params, d, c) for (d, c) in zip(doc_words, doc_counts)])
    return np.mean(np.linalg.norm(thetas - doc_thetas, axis=1))

def document_similarity_matrix(thetas):    
    M = np.zeros((len(thetas), len(thetas)))
    for i in xrange(len(thetas)):
        for j in xrange(len(thetas)):
            M[i, j] = scipy.stats.pearsonr(thetas[i], thetas[j])[0]
            
    return M
    
def cor_mat(sigma):
    inv_sigma = np.linalg.inv(sigma)
    result = np.zeros(sigma.shape)
    for i in xrange(inv_sigma.shape[0]):
        for j in xrange(inv_sigma.shape[1]):
            result[i, j] = np.abs(inv_sigma[i, j])/np.sqrt(inv_sigma[i,i]*inv_sigma[j,j])
    return result

def dsm_rmse(inf, ref):
    return np.sqrt(np.sum(np.square(inf-ref)) / ref.size) / (np.max(ref) - np.min(ref))

def normalize_mu_sigma(mu, sigma):    
    n_samples = 10000
    samples = np.array([f(s) for s in np.random.multivariate_normal(mu, sigma, n_samples)])
    return (np.mean(samples, axis=0), np.cov(samples.T))
    
def validation(doc_words, doc_counts, doc_thetas, voc_len, K):
    corpus = zip(doc_words, doc_counts, doc_thetas)
    np.random.shuffle(corpus)
    
    split = int(len(corpus)*0.8)
    train_data = corpus[:split]
    test_data = corpus[split:]
    
    (train_words, train_counts, train_thetas) = zip(*train_data)

    priors = [np.random.uniform(0, 1, voc_len) for _ in xrange(K)]
    priors = np.array([p / sum(p) for p in priors])
    
    ps = list(expectation_maximization(train_words, train_counts, K, priors))
    m_params = ps[-1][0]
    
    (test_words, test_counts, test_thetas) = zip(*test_data)
    priors = [np.random.uniform(0, 1, voc_len) for _ in xrange(K)]
    priors = np.array([p / sum(p) for p in priors])
        
    train_inf_thetas = np.array([expected_theta(variational_inference(d, c, m_params), m_params, d, c) for (d, c) in zip(train_words, train_counts)])
    test_inf_thetas = np.array([expected_theta(variational_inference(d, c, m_params), m_params, d, c) for (d, c) in zip(test_words, test_counts)])
    
    test_reference = document_similarity_matrix(test_thetas)
    test_inferred = document_similarity_matrix(test_inf_thetas)
    
    train_reference = document_similarity_matrix(train_thetas)
    train_inferred = document_similarity_matrix(train_inf_thetas)
    
    
    print "Reference-inferred RMSE on the train set: %f" % dsm_rmse(train_inferred, train_reference)
    
    print "Reference-inferred RMSE on the test set: %f" % dsm_rmse(test_inferred, test_reference)
    