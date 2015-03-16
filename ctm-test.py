# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

from variational_inference import variational_inference
from expectation_maximization import expectation_maximization
from inference import expected_theta
from evaluation_tools import generate_random_corpus, document_similarity_matrix, dsm_rmse, normalize_mu_sigma, f, cosine_similarity
from math_utils import cor_mat

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.misc
from itertools import groupby
np.set_printoptions(precision=5, linewidth=120)
plt.style.use('ggplot')
import pygraphviz

diss_data_root = "D:\\diss-data\\"
#diss_data_root = "/mnt/B28A02CE8A028ED3/diss-data/"

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
    plt.xlabel("Pathway proportion")
    
def load_evaluation_dataset(supported_pathways=None):
    from collections import defaultdict

    result = defaultdict(list)    
    
    with open(diss_data_root + "CTD_chem_pathways_enriched_for_validation.txt", 'r') as f:
        for line in f.readlines()[1:]:
            tokens = line.strip().lower().split('\t')
            
            drug = tokens[0].replace('"', '')
            pathway = int(tokens[1])
            
            if supported_pathways is None or pathway in supported_pathways:
                result[drug].append(pathway)
                
    return result

def precision(k, rank, reference): 
    return float(len(set(rank[:k]).intersection(reference))) / float(k)

def recall(k, rank, reference):
    return float(len(set(rank[:k]).intersection(reference))) / float(len(reference))

def average_precision(theta, reference):
    rank = sorted(range(len(theta)), key=lambda x: theta[x], reverse=True)

    result = 0    
    
    for k, p in enumerate(rank):
        if p in reference:
            result += precision(k+1, rank, reference)
    
    result /= len(reference)
    return result

def precision_recall(theta, reference):
    rank = sorted(range(len(theta)), key=lambda x: theta[x], reverse=True)
    
    return [(precision(k+1, rank, reference), recall(k+1, rank, reference)) for k in xrange(len(theta))]

def mul_precision_recall(thetas, eval_data):
    ranks = [sorted(range(len(t)), key=lambda x: t[x], reverse=True) for t in thetas]
    
    ref_len = sum([len(e) for e in eval_data])
    
    result = []
    
    for k in xrange(len(thetas[0])):
        pre = float(sum([precision(k+1, r, ref) for r, ref in zip(ranks, eval_data)])) / len(thetas)
        rec = float(sum([recall(k+1, r, ref) * len(ref) for r, ref in zip(ranks, eval_data)])) / ref_len
        result.append((pre, rec))
    
    return result

def evaluate_drug_theta(theta, reference):
    #Return the average theta value for each of the pathways that this drug actually has
    return np.sum([theta[p] for p in reference])

def validate_all_thetas(thetas, eval_data):
    return np.array([evaluate_drug_theta(t, e) for t, e in zip(thetas, eval_data)])

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

def calc_heatmap(thetas, eval_data):
    
    ranks = [sorted(range(len(t)), key=lambda x: t[x], reverse=True) for t in thetas]
    img = np.zeros(thetas.T.shape)
    
    for d, reference, rank in zip(xrange(len(eval_data)), eval_data, ranks):
        for i, pathway in enumerate(rank):
            img[i, d] = 1 if pathway in reference else 0
    
    return img
    
def plot_heatmap(thetas, eval_data):
    imshow(calc_heatmap(thetas, eval_data), cmap="Greys_r", interpolation='nearest')

#Filter the thetas so that they only mention the drugs and the pathways in the reference dataset
def filter_thetas(thetas, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names):
    filtered = np.delete(thetas, [i for i, p in enumerate(pathway_ids) if p not in pathways_in_eval], axis=1)
    filtered /= np.sum(filtered, axis=1)[:, np.newaxis]
    
    filtered = np.delete(filtered, [i for i, d in enumerate(drug_names) if d not in drug_names_in_eval], axis=0)
    
    return filtered

def plot_avg_pr_curve(thetas, reference):
    
    all_precisions = mul_precision_recall(thetas, reference)
    all_precisions.sort(key=lambda x: x[1])
    
    precs, recs = zip(*all_precisions)
    
    plot(recs, precs)

def lda_likelihood(cvt, cdt, phi, theta):
    cdv = cdt.dot(cvt.T)
    return np.multiply(cdv, np.log(theta.dot(phi.T)))

def calc_all_lda_likelihoods():
    lda_theta = np.loadtxt(diss_data_root + "lda/theta.txt")
    lda_phi = np.loadtxt(diss_data_root + "lda/phi.txt")
    
    likelihoods = np.zeros((1000))
    
    for i in xrange(1000):
        cvt = np.loadtxt(diss_data_root + "lda/answers_cvt_%d.txt" % i)
        cdt = np.loadtxt(diss_data_root + "lda/answers_cdt_%d.txt" % i)
        likelihoods[i] = np.sum(lda_likelihood(cvt, cdt, lda_phi, lda_theta))
        
    return likelihoods

def density(M):
    return np.mean((M != 0))

def load_toy_dataset(i):
    data = np.load(diss_data_root + "%d-dataset.npz" % i)
    doc_words = data['arr_0']
    doc_counts = data['arr_1']
    doc_thetas = data['arr_2']
    mu = data['arr_3']
    sigma = data['arr_4']
    beta = data['arr_5']
    
    data = np.load(diss_data_root + "%d-results.npz" % i)
    m_params = data['arr_0'].item()
    thetas = data['arr_1']
    
    return doc_words, doc_counts, doc_thetas, mu, sigma, beta, m_params, thetas

def save_toy_dataset(i, doc_words, doc_counts, doc_thetas, mu, sigma, beta, m_params, thetas):
    np.savez_compressed(diss_data_root + "%d-dataset" % i, doc_words, doc_counts, doc_thetas, mu, sigma, beta)
    np.savez_compressed(diss_data_root + "%d-results" % i, m_params, thetas)
    
#TODO: try the CDF/heatmap/side evaluation on the simulated data
#writeup the simulated study
#send the random thetas
#run exp() with the disease dataset
#try normalizing the logs to 0..inf
#density/performance on simulated

if __name__ == "__main__":
    drug_prune = 1
    gene_prune = 1
    pathway_prune = 1
#
#    drug_gene = np.loadtxt(diss_data_root + "drug_gene_expression_matrix.txt", skiprows=1).T
#    drug_gene = np.exp(drug_gene)
#    drug_gene = drug_gene[::drug_prune,::gene_prune]     
#    
#    doc_words = [d.nonzero()[0] for d in drug_gene]
#    doc_counts = [d[d.nonzero()[0]] for d in drug_gene]
#
#    priors = np.loadtxt(diss_data_root + "gene_pathway_matrix_K.txt", skiprows=1).T[::pathway_prune,::gene_prune]
#    priors = np.array([p / sum(p) for p in priors])
#    beta_density = density(priors) #0.0138 for ctd
#
#    print "Drugs: %d, pathways: %d, genes: %d" % (drug_gene.shape[0], priors.shape[0], drug_gene.shape[1])
### 
##    m_params, v_params = expectation_maximization(doc_words, doc_counts, len(priors), priors, max_iterations=100)
#    
###    
####    
    data = np.load(diss_data_root + "../data-every-1th-drug-exp.npz")
    m_params = data['arr_0'].item()
###    v_params = data['arr_1']
###    
    data = np.load(diss_data_root + "../thetas-every-1th-drug-exp.npz")
    thetas = data['arr_0']
##  
##    thetas = np.array([expected_theta(v, m_params, w, c) for v, w, c in zip(v_params, doc_words, doc_counts)])      
##    
##    
    pathway_ids = [int(p.strip()) for p in open(diss_data_root + "pathway_id.txt").readlines()][::pathway_prune]
    pathway_names = [p.strip() for p in open(diss_data_root + "pathway_names_used.txt").readlines()][::pathway_prune]
    drug_names = [d.strip() for d in open(diss_data_root + "drug_name.txt").readlines()][1::drug_prune]
    
    eval_data = load_evaluation_dataset(set(pathway_ids))
    pathway_map = {v: k for k, v in enumerate(pathway_ids)}
    pathways_in_eval = set.union(*([set(eval_data[d]) for d in drug_names if d in eval_data]))
    pathway_ids_in_eval = sorted(pathways_in_eval)
    pathway_map_in_eval = {v: k for k, v in enumerate(pathway_ids_in_eval)}
    drug_names_in_eval = [d for d in drug_names if d in eval_data]
    eval_data = [[pathway_map_in_eval[p] for p in eval_data[d]] for d in drug_names_in_eval]
#
#    
####    
####            
#Load the LDA phi values and transpose to have the same shape as our beta
    lda_phi = np.loadtxt(diss_data_root + "lda-phi-paper.txt").T
    lda_phi /= np.sum(lda_phi, axis=1)[:, np.newaxis]
####    
##    
#    
#    
#    
##
    lda_thetas = np.loadtxt(diss_data_root + "lda-theta-paper.txt")
    lda_thetas_f = filter_thetas(lda_thetas, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names)
##    
    ctm_thetas_f = filter_thetas(thetas, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names)
##
    random_thetas = np.random.uniform(size=ctm_thetas_f.shape)
    random_thetas /= np.sum(random_thetas, axis=1)[:, np.newaxis]

#    #Construct thetas implied by the evaluation data
    eval_thetas = np.zeros((len(drug_names_in_eval), len(pathway_names)))
    
    for i, e in enumerate(eval_data):
        for p in e:
            eval_thetas[i, p] = 1
            
    eval_thetas /= np.sum(eval_thetas, axis=1)[:, np.newaxis]
    theta_sparsity = density(eval_thetas) #0.1231 for ctd dataset

    
    gmrf_thetas = filter_thetas(scipy.io.loadmat(diss_data_root + "output_s_me.mat")['output_S'].T, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names)
    gmrf_thetas /= np.sum(gmrf_thetas, axis=1)[:, np.newaxis]
##    
##    lda_perf = validate_all_thetas(drug_names, lda_thetas_f, eval_data, pathway_map_in_eval)
#    ctm_perf = validate_all_thetas(ctm_thetas_f, eval_data)
#    random_perf = validate_all_thetas(random_thetas, eval_data)
#    gmrf_perf = validate_all_thetas(gmrf_thetas, eval_data)
##    
#    ctm_side = np.sum(calc_heatmap(ctm_thetas_f, eval_data), axis=1)
#    gmrf_side = np.sum(calc_heatmap(gmrf_thetas, eval_data), axis=1)
##    lda_side = np.sum(calc_heatmap(drug_names, pathway_ids_in_eval, lda_thetas_f, eval_data), axis=1)
#    ran_side = np.sum(calc_heatmap(random_thetas, eval_data), axis=1)
#    ref_side = np.sum(calc_heatmap(eval_thetas, eval_data), axis=1)
#    
#    map(plot_cdf, [ctm_perf, random_perf, gmrf_perf])
#    legend(['CTM', 'Random'])
#    title('CDF of the proportion of drug-pathway perturbations explained by each model')
#    
#    ctm_ap = [average_precision(t, e) for t, e in zip (ctm_thetas_f, eval_data)]
#    
#    ctm_pr = [(p, r) for (p, r) in precision_recall(t, e) for t, e in zip(ctm_thetas_f, eval_data)]
#    ctm_pr.sort(key=lambda x: x[1])
#    
#    precisions = [(r, list(ps)) for r, ps in groupby(ctm_pr, lambda x: x[1])]
#    precisions = [(r, np.mean(ps), np.std(ps)) for r, ps in precisions]
#    recs, precmeans, precstds = zip(*precisions)
#
#    
##    
##    mu_n, sigma_n = normalize_mu_sigma(m_params.mu, m_params.sigma)
##    stdevs = np.sqrt(sigma_n.diagonal())
##    thetas_norm = [(t - mu_n)/stdevs for t in thetas]
#    
#    
##    
##    f = open("D:\\drug-mparams-every-" + str(drug_prune) + "th-drug-exp.npz", 'wb')
##    np.savez_compressed(f, m_params)
##    f.close()
##    
##    f = open("D:\\drug-thetas-every-" + str(drug_prune) + "th-drug-exp.npz", 'wb')
##    np.savez(f, thetas)
##    f.close()
    
        
#TODO:
#manually assign the parameters (sigma, mu)
#gibbs sampling for logistic normal topic models with graph-based priors    
#Evaluation: RMSE for theta, beta, corr mat, visual graph (threshold for corr)
#vary K, sparsity of beta(how many zeros in each topic)/topic graph
#check out bdgraph!!!!
#def generated_corpus_evaluation():
    voc_len = 3000
    K = 50
    N_d = 1000
    no_docs = 500
    
    mu = np.loadtxt(diss_data_root + "ctd-implied-mu.txt")
    sigma = np.loadtxt(diss_data_root + "ctd-implied-sigma.txt")
    
    print "Generating a random corpus..."
    doc_words, doc_counts, doc_thetas, mu, sigma, beta = generate_random_corpus(voc_len, K, N_d, no_docs, 0.1231, 0.0138)
    
#    validation(doc_words, doc_counts, doc_thetas)
    
    priors = np.ones((K, voc_len))
    priors[beta == 0] = 0

    print "Performing expectation maximization..."    
    priors = np.array([p / sum(p) for p in priors])
    m_params, v_params = expectation_maximization(doc_words, doc_counts, K, priors, max_iterations=100)
    
    corr = document_similarity_matrix(beta, m_params.beta)
#    
#    print "Reference-inferred beta similarity matrix (for topic identifiability):"
#    print corr
    
    print "Evaluating by classifying the training dataset..."
    thetas = np.array([expected_theta(v, m_params, w, c) for v, w, c in zip(v_params, doc_words, doc_counts)])
        
    permuted_thetas = np.array(thetas)
    permuted_mu = np.array(m_params.mu)    
    permuted_sigma = np.array(m_params.sigma)
    permuted_beta = np.array(m_params.beta)
    
    #todo: axis=1 gives better results
    betamap = np.argmax(corr, axis=0) #betamap[i] : inferred topic id that's most similar to the actual topic i
    if (len(np.unique(betamap)) < len(betamap)):
        print "Warning: could not infer the topic mapping, topics not far enough apart"
    else:
        for d in xrange(len(thetas)):
            for i in xrange(len(betamap)):
                permuted_thetas[d, betamap[i]] = thetas[d, i]
        for i in xrange(len(mu)):
            permuted_mu[betamap[i]] = m_params.mu[i]
        for i in xrange(K):
            for j in xrange(K):
                permuted_sigma[betamap[i], betamap[j]] = m_params.sigma[i, j]
        for i in xrange(K):
            permuted_beta[betamap[i]] = m_params.beta[i]
    
    theta_diff_sizes = [dsm_rmse(inf, ref) for inf, ref in zip(permuted_thetas, doc_thetas)]
    
    baseline = np.random.multivariate_normal(mu, sigma, size=no_docs)
    baseline = np.exp(baseline) / np.sum(np.exp(baseline), axis=1)[:, None]
    baseline_diff_sizes = [cosine_similarity(inf, ref) for inf, ref in zip(baseline, doc_thetas)]
    
    baseline2 = np.random.uniform(size=baseline.shape)
    baseline2 = np.exp(baseline2) / np.sum(np.exp(baseline2), axis=1)[:, None]
    baseline2_diff_sizes = [cosine_similarity(inf, ref) for inf, ref in zip(baseline2, doc_thetas)]
    
    ref_ranks = [np.where(d > 0)[0] for i, d in enumerate(doc_thetas)]
    
    thetas = np.array(thetas)
    doc_thetas = np.array(doc_thetas)
    
    ctm_perf = validate_all_thetas(thetas, ref_ranks)
    rnd_perf = validate_all_thetas(baseline2, ref_ranks)
    ref_perf = validate_all_thetas(doc_thetas, ref_ranks)
    
    ctm_side = np.sum(calc_heatmap(thetas, ref_ranks), axis=1)
    ref_side = np.sum(calc_heatmap(doc_thetas, ref_ranks), axis=1)
    rnd_side = np.sum(calc_heatmap(baseline2, ref_ranks), axis=1)
    
    ctm_ap = [average_precision(t, r) for t, r in zip(thetas, ref_ranks)]
    rnd_ap = [average_precision(t, r) for t, r in zip(baseline2, ref_ranks)]
    ref_ap = [average_precision(t, r) for t, r in zip(doc_thetas, ref_ranks)]

#    
#    plot_cdf(theta_diff_sizes)
#    plot_cdf(baseline2_diff_sizes)
#    
#    plt.legend([r"Inferred $\theta$", r"Baseline $\theta$ (random)"])
#    plt.xlabel("RMSE $\\theta_{inf}$ from $\\theta_{ref}$")
#    plt.ylabel("Proportion of errors below a given RMSE (the CDF)")
#        
    reference = document_similarity_matrix(doc_thetas)
    inferred = document_similarity_matrix(thetas)
    
    print "RMSE between inferred document correlations and the reference: %f" % dsm_rmse(inferred, reference)
    print "RMSE between inferred beta and the reference: %f" % dsm_rmse(np.array(permuted_beta), np.array(beta))
    print "RMSE between inferred topic correlations and the reference: %f" % dsm_rmse(cor_mat(permuted_sigma), cor_mat(sigma))
    print "RMSE between inferred topic proportions and the reference: %f" % dsm_rmse(f(permuted_mu), f(mu))
    
        
    
#    