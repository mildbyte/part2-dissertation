# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

from variational_inference import variational_inference
from expectation_maximization import expectation_maximization
from inference import expected_theta
from evaluation_tools import generate_random_corpus, document_similarity_matrix, dsm_rmse
from math_utils import cor_mat

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.misc
import networkx as nx
np.set_printoptions(precision=2, linewidth=120)

def plot_correlations(sigma, threshold=0.1):
    corr = cor_mat(sigma)
    K = corr.shape[0]
    
    G = nx.Graph()
    G.add_nodes_from(xrange(K))
    G.add_weighted_edges_from([(a, b, 1.0/corr[a, b]) for a in xrange(K) for b in xrange(K) if a != b and corr[a, b] > threshold])
    
    pos = nx.fruchterman_reingold_layout(G)
    nx.draw(G, pos, with_labels=True, edge_cmap=plt.get_cmap("Blues"), alpha=0.7, node_size=500, width=5.0, labels={i: i for i in xrange(K)})
    labels={(u,v,) : "%.2f" % (1.0/d['weight']) for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

def plot_cdf(arr):
    counts, edges = np.histogram(arr, normed=True, bins=1000)
    cdf = np.cumsum(counts)
    cdf /= max(cdf)
    plt.plot(edges[1:], cdf)
    
if __name__ == "__main__":
    drug_prune = 10
    gene_prune = 10
    pathway_prune = 10
    
    drug_gene = np.loadtxt("D:\\diss-data\\gene_expression_matrix_X.txt").T    
    drug_gene = np.round(np.abs(drug_gene*100)).astype('int')
    drug_gene = drug_gene[::drug_prune,::gene_prune]        
    
    doc_words = [d.nonzero()[0] for d in drug_gene]
    doc_counts = [d[d.nonzero()[0]] for d in drug_gene]
#
#    pathway_gene = np.loadtxt("D:\\diss-data\\gene_pathway_matrix_K.txt")[::pathway_prune,::gene_prune]
#    priors = np.multiply(np.random.uniform(size=pathway_gene.shape), pathway_gene)
#    #priors = np.random.uniform(size=pathway_gene.shape)
#    #priors[priors > 0.5] = 0
#    #priors /= 0.5
#    
#    print "Drugs: %d, pathways: %d, genes: %d" % (drug_gene.shape[0], pathway_gene.shape[0], drug_gene.shape[1])
#    priors = np.array([p / sum(p) for p in priors])
#    
#    m_params, v_params = expectation_maximization(doc_words, doc_counts, len(priors), priors)
    
    ps = list(expectation_maximization(doc_words, doc_counts, len(priors), priors))
    t = expected_theta(v_params[0], m_params, doc_words[0], doc_counts[0])

        
#TODO:
#manually assign the parameters (sigma, mu)
#gibbs sampling for logistic normal topic models with graph-based priors    
#Evaluation: RMSE for theta, beta, corr mat, visual graph (threshold for corr)
#vary K, sparsity of beta(how many zeros in each topic)/topic graph
#see the GMRF paper for toy dataset generation
#check out bdgraph
    
#    voc_len = 3000
#    K = 260
#    N_d = 10
#    no_docs = 1200
#    
#    print "Generating a random corpus..."
#    doc_words, doc_counts, doc_thetas, mu, sigma, beta = generate_random_corpus(voc_len, K, N_d, no_docs)
#    
##    validation(doc_words, doc_counts, doc_thetas)
#    
#    
#    priors = [np.random.uniform(0, 1, voc_len) for _ in xrange(K)]
#    for i in xrange(K):
#        priors[i][i] = 0
#
#    print "Performing expectation maximization..."    
#    priors = np.array([p / sum(p) for p in priors])
#    ps = list(expectation_maximization(doc_words, doc_counts, K, priors))
#    
#    m_params = ps[-1][0]
#    
#    
#    diff = np.zeros((K, K))
#    for i in xrange(K):
#        for j in xrange(K):
#            diff[i, j] = np.linalg.norm(beta[i] - m_params.beta[j])
#            
#    corr = np.zeros((K, K))
#    for i in xrange(K):
#        for j in xrange(K):
#            corr[i, j] = scipy.stats.pearsonr(beta[i], m_params.beta[j])[0]
#            
##    Setting diagonal entries to 0 in the prior for identifiability
##       => don't need to permute the results
##    
##    betamap = np.argmax(corr, axis=0) #betamap[i] : inferred topic id that's most similar to the actual topic i
##    if (len(np.unique(betamap)) < len(betamap)):
##        print "Warning: could not infer the topic mapping, topics not far enough apart"
##        print diff
##    
#    print "Evaluating by classifying the training dataset..."
#    thetas = np.array([expected_theta(variational_inference(d, c, m_params), m_params, d, c) for (d, c) in zip(doc_words, doc_counts)])
##    permuted_thetas = np.array(thetas)
##    for d in xrange(len(thetas)):
##        for i in xrange(len(betamap)):
##            permuted_thetas[d, betamap[i]] = thetas[d, i]
#    
#    theta_diff_sizes = [dsm_rmse(inf, ref) for inf, ref in zip(thetas, doc_thetas)]
#    
#    baseline = np.random.multivariate_normal(mu, sigma, size=no_docs)
#    baseline = np.exp(baseline) / np.sum(np.exp(baseline), axis=1)[:, None]
#    baseline_diff_sizes = [dsm_rmse(inf, ref) for inf, ref in zip(baseline, doc_thetas)]
#    
#    plot_cdf(theta_diff_sizes)
#    plot_cdf(baseline_diff_sizes)
#    
#    plt.legend([r"Inferred $\theta$", r"Baseline $\theta$ (random)"])
#    plt.xlabel("RMSE $\\theta_{inf}$ from $\\theta_{ref}$")
#    plt.ylabel("Proportion of errors below a given RMSE (the CDF)")
#        
#    reference = document_similarity_matrix(doc_thetas)
#    inferred = document_similarity_matrix(thetas)
#    
#    print "RMSE between inferred document correlations and the reference: %f" % dsm_rmse(inferred, reference)
#    
    
