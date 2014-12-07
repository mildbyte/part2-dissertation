# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

from variational_inference import variational_inference
from expectation_maximization import expectation_maximization
from inference import expected_theta
from evaluation_tools import generate_random_corpus, document_similarity_matrix, dsm_rmse

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.misc
np.set_printoptions(precision=2, linewidth=120)

def plot_cdf(arr):
    counts, edges = np.histogram(arr, normed=True, bins=1000)
    cdf = np.cumsum(counts)
    cdf /= max(cdf)
    plt.plot(edges[1:], cdf)

if __name__ == "__main__":
    #Final data: 6101 drugs (documents), 22283 genes (words), 260 pathways (topics)
        
#TODO:
#manually assign the parameters (sigma, mu)
#topic covariances
#K-1 instead of k
#gibbs sampling for logistic normal topic models with graph-based priors
#compare sigma, mu by drawing many etas, normalizing them and comparing to the reference etas, t-test!
#find a way to evaluate sigma/mu (multinomial statistics?)
#wishart to get sigma + generate from an adjacency matrix for a graph
    
#Evaluation: RMSE for theta, beta, corr mat, visual graph (threshold for corr)
    #vary K, sparsity of beta(how many zeros in each topic)/topic graph
    #see the GMRF paper for toy dataset generation
    #check out bdgraph
    
    voc_len = 10
    K = 2
    N_d = 100
    no_docs = 256
    
    doc_words, doc_counts, doc_thetas, mu, sigma, beta = generate_random_corpus(voc_len, K, N_d, no_docs)
    
#    validation(doc_words, doc_counts, doc_thetas)
    
    
    priors = [np.random.uniform(0, 1, voc_len) for _ in xrange(K)]
    for i in xrange(K):
        priors[i][i] = 0
    
    priors = np.array([p / sum(p) for p in priors])
    ps = list(expectation_maximization(doc_words, doc_counts, K, priors))
    
    m_params = ps[-1][0]
    
    
    diff = np.zeros((K, K))
    for i in xrange(K):
        for j in xrange(K):
            diff[i, j] = np.linalg.norm(beta[i] - m_params.beta[j])
            
    corr = np.zeros((K, K))
    for i in xrange(K):
        for j in xrange(K):
            corr[i, j] = scipy.stats.pearsonr(beta[i], m_params.beta[j])[0]
    
    
    betamap = np.argmax(corr, axis=0) #betamap[i] : inferred topic id that's most similar to the actual topic i
    if (len(np.unique(betamap)) < len(betamap)):
        print "Warning: could not infer the topic mapping, topics not far enough apart"
        print diff
    
    thetas = np.array([expected_theta(variational_inference(d, c, m_params), m_params, d, c) for (d, c) in zip(doc_words, doc_counts)])
    permuted_thetas = np.array(thetas)
    for d in xrange(len(thetas)):
        for i in xrange(len(betamap)):
            permuted_thetas[d, betamap[i]] = thetas[d, i]
    
    theta_differences = permuted_thetas - doc_thetas
    theta_diff_sizes = np.linalg.norm(theta_differences, axis=1)
    
    baseline = np.random.multivariate_normal(mu, sigma, size=no_docs)
    baseline = np.exp(baseline) / np.sum(np.exp(baseline), axis=1)[:, None]
    baseline_diff = baseline - doc_thetas
    baseline_diff_sizes = np.linalg.norm(baseline_diff, axis=1)
    
    plot_cdf(theta_diff_sizes)
    plot_cdf(baseline_diff_sizes)
    
    plt.legend([r"Inferred $\theta$", r"Baseline $\theta$ (random)"])
    plt.xlabel("Norm of $\\theta_{inf}$ - $\\theta_{ref}$")
    plt.ylabel("Proportion of errors below a given norm (the CDF)")
    
        
    reference = document_similarity_matrix(doc_thetas)
    inferred = document_similarity_matrix(thetas)
    
    print "RMSE between inferred document correlations and the reference: %f" % dsm_rmse(inferred, reference)
    