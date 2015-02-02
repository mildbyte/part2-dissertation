# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

from variational_inference import variational_inference
from expectation_maximization import expectation_maximization
from inference import expected_theta
from evaluation_tools import generate_random_corpus, document_similarity_matrix, dsm_rmse, normalize_mu_sigma, f
from math_utils import cor_mat

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.misc
np.set_printoptions(precision=2, linewidth=120)
plt.style.use('ggplot')
import pygraphviz

#diss_data_root = "D:\\diss-data\\"
diss_data_root = "/mnt/B28A02CE8A028ED3/diss-data/"

def connected_subgraph(G, node, max_depth=3):
    nodes = set([node])
    depth = 0
    
    G2 = pygraphviz.AGraph('graph G {}')
    G2.node_attr['shape'] = 'box'
    G2.graph_attr['splines'] = 'spline'
    G2.graph_attr['overlap'] = 'prism'    
    
    while True:
        depth += 1
        if depth == max_depth:
            break
        
        newnodes = set(nodes)
        for n in nodes:
            for e in G.iteredges(n):
                G2.add_edge(e, attr=e.attr)
            newnodes.update(G.iterneighbors(n))
        
        if newnodes == nodes:
            break           
        
        nodes = newnodes
    
    return G2
    
def generate_graph(corr, threshold=0.1, sigma_labels=None):    
    K = corr.shape[0]
    if sigma_labels is None:
        sigma_labels = xrange(K)
        
    G = pygraphviz.AGraph('graph G {}')
    G.node_attr['shape'] = 'box'
    G.graph_attr['splines'] = 'spline'
    G.graph_attr['overlap'] = 'prism'
    
    for a in xrange(K):
        for b in xrange(a):
            if corr[a, b] > threshold:
                G.add_edge(sigma_labels[a], sigma_labels[b], len=(1.0 - corr[a, b]), label="%.2f" % corr[a, b])
    
    degree = G.degree()
    G.delete_nodes_from([n for n in degree if degree[n] == 0])

    return G

def plot_correlations(G):
    G.draw(path="pathways.png", prog='sfdp', args='-Gdpi=200')

def plot_cdf(arr):
    counts, edges = np.histogram(arr, normed=True, bins=1000)
    cdf = np.cumsum(counts)
    cdf /= max(cdf)
    plt.plot(edges[1:], cdf)

def plot_rank(theta, pathway_labels=None, prune_top=None, drug_name=None):
    if prune_top is None:
        prune_top = len(theta)
    
    sorted_pathways = range(len(theta))
    sorted_pathways.sort(key=lambda p: theta[p], reverse=True)
    
    if pathway_labels is None:
        pathway_labels = sorted_pathways[:prune_top]
    else:
        pathway_labels = np.array(pathway_labels)[sorted_pathways[:prune_top]]
    
    patches = plt.barh(range(prune_top), theta[sorted_pathways[:prune_top]])
    [plt.text(p.xy[0] + 0.5 * p.get_width(), p.xy[1] + 0.5 * p.get_height(), l, ha='center', va='center') for p, l in zip(patches, pathway_labels)]
    plt.yticks(np.arange(prune_top), np.arange(prune_top), va='center')
    
    if drug_name is not None:
        plt.title(drug_name)
    
    plt.ylabel("Pathway rank")    
    plt.xlabel("Pathway proportion, $\\sigma$s from the mean")
    
def load_evaluation_dataset(supported_pathways):
    from collections import defaultdict

    result = defaultdict(list)    
    
    with open(diss_data_root + "CTD_chem_pathways_enriched_for_validation.txt", 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().lower().split('\t')
            
            drug = tokens[0].replace('"', '')
            pathway = int(tokens[1])
            
            if pathway in supported_pathways:
                result[drug].append(pathway)
    
    return result

def evaluate_drug_theta(theta, pathway_map, reference_pathways):
    #Return the average theta value for each of the pathways that this drug actually has
    return np.sum([theta[pathway_map[p]] for p in reference_pathways])

def validate_all_thetas(drug_names, thetas, eval_data, pathway_map):
    return np.array([evaluate_drug_theta(t, pathway_map, eval_data[d]) for d, t in zip(drug_names, thetas) if d in eval_data])

#Performs an evaluation cycle for given topic and vocabulary length, returning the performance measure
def generated_rmse_evaluation(K, voc_len):
    N_d = 100
    no_docs = 100
    doc_words, doc_counts, doc_thetas, mu, sigma, beta = generate_random_corpus(voc_len, K, N_d, no_docs)
    
    priors = [np.ones(voc_len) for _ in xrange(K)]
    for i in xrange(max([K, voc_len])):
        priors[i % K][i % voc_len] = 0
    priors = np.array([p / sum(p) for p in priors])
    m_params, v_params = expectation_maximization(doc_words, doc_counts, K, priors, max_iterations=100)
    
    thetas = np.array([expected_theta(v, m_params, w, c) for v, w, c in zip(v_params, doc_words, doc_counts)])
    reference = document_similarity_matrix(doc_thetas)
    inferred = document_similarity_matrix(thetas)
    
    return dsm_rmse(inferred, reference)
    
def most_similar_drug_ids(sim_matrix, drug_id):
    return sorted(range(len(sim_matrix)), key=lambda x: sim_matrix[drug_id, x], reverse=True)

if __name__ == "__main__":
    drug_prune = 1
    gene_prune = 1
    pathway_prune = 1

    
    drug_gene = np.loadtxt(diss_data_root + "gene_expression_matrix_X.txt").T    
    drug_gene = np.round(np.abs(drug_gene*100)).astype('int')
    drug_gene = drug_gene[::drug_prune,::gene_prune]     
    
    doc_words = [d.nonzero()[0] for d in drug_gene]
    doc_counts = [d[d.nonzero()[0]] for d in drug_gene]

    priors = np.loadtxt(diss_data_root + "gene_pathway_matrix_K.txt")[::pathway_prune,::gene_prune]
    priors = np.array([p / sum(p) for p in priors])
    print "Drugs: %d, pathways: %d, genes: %d" % (drug_gene.shape[0], priors.shape[0], drug_gene.shape[1])
# 
#    m_params, v_params = expectation_maximization(doc_words, doc_counts, len(priors), priors, max_iterations=100)
#    thetas = parallel_expected_theta(v_params, m_params, doc_words, doc_counts)

    
    data = np.load(diss_data_root + "../data-every-1th-drug.npz")
    m_params = data['arr_0'].item()
    #v_params = data['arr_1']
    
    data = np.load(diss_data_root + "../thetas-every-1th-drug.npz")
    thetas = data['arr_0']
        
    pathway_ids = [int(p.strip()) for p in open(diss_data_root + "pathway_id.txt").readlines()][::pathway_prune]
    pathway_names = [p.strip() for p in open(diss_data_root + "pathway_names_used.txt").readlines()][::pathway_prune]
    drug_names = [d.strip() for d in open(diss_data_root + "drug_name.txt").readlines()][1::drug_prune]
    eval_data = load_evaluation_dataset(set(pathway_ids))
    pathway_map = {v: k for k, v in enumerate(pathway_ids)}
    
    #Construct thetas implied by the evaluation data
    drug_names_in_eval = [d for d in drug_names if d in eval_data]
    eval_thetas = np.zeros((len(drug_names_in_eval), len(pathway_names)))
    
    for i, d in enumerate(drug_names_in_eval):
        for p in eval_data[d]:    
            eval_thetas[i, pathway_map[p]] = 1
            
    eval_thetas /= np.sum(eval_thetas, axis=1)[:, np.newaxis]
            
    #Load the LDA phi values and transpose to have the same shape as our beta
    lda_phi = np.loadtxt(diss_data_root + "lda-phi.txt").T
    lda_phi /= np.sum(lda_phi, axis=1)[:, np.newaxis]
    
    lda_thetas = np.loadtxt(diss_data_root + "lda-theta.txt")
    lda_thetas /= np.sum(lda_thetas, axis=1)[:, np.newaxis]
    
    random_thetas = np.random.uniform(size=lda_thetas.shape)
    random_thetas /= np.sum(random_thetas, axis=1)[:, np.newaxis]
    
    random_ctm_thetas = np.random.multivariate_normal(m_params.mu, m_params.sigma, size=len(drug_names))
    random_ctm_thetas = np.exp(random_ctm_thetas) / np.sum(np.exp(random_ctm_thetas), axis=1)[:, None]
    
    lda_perf = validate_all_thetas(drug_names, lda_thetas, eval_data, pathway_map)
    ctm_perf = validate_all_thetas(drug_names, thetas, eval_data, pathway_map)
    random_perf = validate_all_thetas(drug_names, random_thetas, eval_data, pathway_map)
    random_ctm_perf = validate_all_thetas(drug_names, random_ctm_thetas, eval_data, pathway_map)
    
    map(plot_cdf, [lda_perf, ctm_perf, random_perf, random_ctm_perf])
    legend(['LDA', 'CTM', 'Random', 'Random with CTM means'])
    title('CDF of the proportion of drug-pathway perturbations explained by each model')
    
    
#    mu_n, sigma_n = normalize_mu_sigma(m_params.mu, m_params.sigma)
#    thetas_norm = [(t - mu_n)/np.sqrt(s) for t, s in zip(thetas, sigma_n.diagonal())]
    
#    
#    
#    f = open("D:\\data-every-" + str(drug_prune) + "th-drug.npz", 'wb')
#    np.savez(f, m_params, v_params)
#    f.close()
#    
#    f = open("D:\\thetas-every-" + str(drug_prune) + "th-drug.npz", 'wb')
#    np.savez(f, thetas)
#    f.close()
    
        
#TODO:
#manually assign the parameters (sigma, mu)
#gibbs sampling for logistic normal topic models with graph-based priors    
#Evaluation: RMSE for theta, beta, corr mat, visual graph (threshold for corr)
#vary K, sparsity of beta(how many zeros in each topic)/topic graph
#see the GMRF paper for toy dataset generation
#check out bdgraph
#def generated_corpus_evaluation():
#    voc_len = 10
#    K = 2
#    N_d = 100
#    no_docs = 1000
#    
#    print "Generating a random corpus..."
#    doc_words, doc_counts, doc_thetas, mu, sigma, beta = generate_random_corpus(voc_len, K, N_d, no_docs)
#    
##    validation(doc_words, doc_counts, doc_thetas)
#    
#    priors = [np.ones(voc_len) for _ in xrange(K)]
#    for i in xrange(max([K, voc_len])):
#        priors[i % K][i % voc_len] = 0
#
#    print "Performing expectation maximization..."    
#    priors = np.array([p / sum(p) for p in priors])
#    m_params, v_params = expectation_maximization(doc_words, doc_counts, K, priors, max_iterations=100)
#    
#    corr = document_similarity_matrix(beta, m_params.beta)
#    
#    print "Reference-inferred beta similarity matrix (for topic identifiability):"
#    print corr
#    
#    print "Evaluating by classifying the training dataset..."
#    thetas = np.array([expected_theta(v, m_params, w, c) for v, w, c in zip(v_params, doc_words, doc_counts)])
#        
#    permuted_thetas = np.array(thetas)
#    permuted_mu = np.array(m_params.mu)    
#    permuted_sigma = np.array(m_params.sigma)
#    permuted_beta = np.array(m_params.beta)
#    
#    betamap = np.argmax(corr, axis=0) #betamap[i] : inferred topic id that's most similar to the actual topic i
#    if (len(np.unique(betamap)) < len(betamap)):
#        print "Warning: could not infer the topic mapping, topics not far enough apart"
#    else:
#        for d in xrange(len(thetas)):
#            for i in xrange(len(betamap)):
#                permuted_thetas[d, betamap[i]] = thetas[d, i]
#        for i in xrange(len(mu)):
#            permuted_mu[betamap[i]] = m_params.mu[i]
#        for i in xrange(K):
#            for j in xrange(K):
#                permuted_sigma[betamap[i], betamap[j]] = m_params.sigma[i, j]
#        for i in xrange(K):
#            permuted_beta[betamap[i]] = m_params.beta[i]
#    
#    theta_diff_sizes = [dsm_rmse(inf, ref) for inf, ref in zip(permuted_thetas, doc_thetas)]
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
#    print "RMSE between inferred beta and the reference: %f" % dsm_rmse(np.array(permuted_beta), np.array(beta))
#    print "RMSE between inferred topic correlations and the reference: %f" % dsm_rmse(cor_mat(permuted_sigma), cor_mat(sigma))
#    print "RMSE between inferred topic proportions and the reference: %f" % dsm_rmse(f(permuted_mu), f(mu))
#    
#        
#    
    
