# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats
from variational_inference import variational_inference
from inference import expected_theta
from collections import Counter
     
def exp_normalise(eta):
    return np.exp(eta) / np.sum(np.exp(eta))
    
def generate_random_corpus(voc_len, K, N_d, no_docs, theta_density=1.0, beta_density=1.0, mu=None, sigma=None):
    def gendoc(mu, sigma, beta):
        density = int(np.clip(np.random.poisson(theta_density * K), 1, K))
        
        eta_d = np.random.multivariate_normal(mu, sigma)
        eta_d = exp_normalise(eta_d)
        eta_d[np.argsort(eta_d)[density:]] = 0
        eta_d /= np.sum(eta_d)
        
        document = []
        topics = np.random.multinomial(N_d, eta_d)
        
        for t, n in enumerate(topics):
            words = beta[t]
            
            W_dn = np.random.choice(xrange(len(words)), size=n, p=words)
            document.extend(W_dn)
    
        return document, eta_d
    
    if mu is None:
        mu = np.random.uniform(0, 1, K)
    
    if sigma is None:
        sigma = sample_wishart(K, np.identity(K) / float(K*K)) #so the base variance for, say, 5 topics will be 0.04
    
    beta = [np.random.uniform(0, 1, voc_len) for _ in xrange(K)]
    for i in xrange(K):
        density = np.clip(np.random.poisson(beta_density * voc_len), 1, voc_len)
        beta[i][np.argsort(beta[i])[density:]] = 0
    beta = np.array([b / sum(b) for b in beta])
    
    doc_words = []
    doc_counts = []
    doc_thetas = []

    for _ in xrange(no_docs):
        doc, eta_d = gendoc(mu, sigma, beta)
        c = Counter(doc)
        
        doc_words.append(np.array(list(c.iterkeys())))
        doc_counts.append(np.array(list(c.itervalues())))
        doc_thetas.append(eta_d)

    return doc_words, doc_counts, doc_thetas, mu, sigma, beta

def error_measure(m_params, doc_words, doc_counts, doc_thetas):
    thetas = np.array([expected_theta(variational_inference(d, c, m_params), m_params, d, c) for (d, c) in zip(doc_words, doc_counts)])
    return np.mean(np.linalg.norm(thetas - doc_thetas, axis=1))

def cosine_similarity(a, b=None):
    return a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b)

def document_similarity_matrix(thetas, thetas2=None):
    if thetas2 is None:
        thetas2 = thetas
        
    M = np.zeros((len(thetas), len(thetas2)))
    for i in xrange(len(thetas)):
        for j in xrange(len(thetas2)):
            M[i, j] = cosine_similarity(thetas[i], thetas2[j])
            
    return M

def sample_wishart(dof, scale):
    cholesky = np.linalg.cholesky(scale)
    
    d = scale.shape[0]
    a = np.zeros((d, d))
    for i in xrange(d):
        for j in xrange(i + 1):
            if i == j:
                a[i, j] = np.sqrt(scipy.stats.chi2.rvs(dof + 2 - i))
            else:
                a[i, j] = np.random.normal(0, 1)
    return cholesky.dot(a).dot(a.T).dot(cholesky.T)
    
def dsm_rmse(inf, ref):
    return np.sqrt(np.sum(np.square(inf-ref)) / ref.size) / (np.max(ref) - np.min(ref))

def normalize_mu_sigma(mu, sigma):    
    n_samples = 10000
    samples = np.array([exp_normalise(s) for s in np.random.multivariate_normal(mu, sigma, n_samples)])
    return (np.mean(samples, axis=0), np.cov(samples.T))