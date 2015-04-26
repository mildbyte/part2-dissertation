# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 10:43:11 2014

@author: mildbyte
"""

from variational_inference import variational_inference
from expectation_maximization import expectation_maximization
from inference import expected_theta, parallel_expected_theta
from evaluation_tools import generate_random_corpus, document_similarity_matrix, dsm_rmse, normalize_mu_sigma, exp_normalise, cosine_similarity
from math_utils import cor_mat
from visualisation_tools import *

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import scipy.misc
import pandas as pd

from scipy.stats import spearmanr, pearsonr

from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation

np.set_printoptions(precision=5, linewidth=120)
#plt.style.use('ggplot')

diss_data_root = "D:\\diss-data\\"
#diss_data_root = "/mnt/B28A02CE8A028ED3/diss-data/"

def generate_random_beta(ref_beta):
    result = np.random.uniform(size=ref_beta.shape)
    result[ref_beta == 0] = 0
    result /= np.sum(result, axis=1)[:, np.newaxis]
    
    return result

def generate_random_theta(ref_theta):
    result = np.random.uniform(size=ref_theta.shape)
    result = np.exp(result) / np.sum(np.exp(result), axis=1)[:, None]
    return result
    
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

def mul_precision_rank(thetas, eval_data):
    ranks = [sorted(range(len(t)), key=lambda x: t[x], reverse=True) for t in thetas]
    
    result = []
    
    for k in xrange(len(thetas[0])):
        pre = float(sum([precision(k+1, r, ref) for r, ref in zip(ranks, eval_data)])) / len(thetas)
        result.append((k, pre))
    
    return result

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
    return np.sum([theta[p] for p in reference]) / len(reference) * len(theta)

def validate_all_thetas(thetas, eval_data):
    return np.array([evaluate_drug_theta(t, e) for t, e in zip(thetas, eval_data)])
    
def most_similar_drug_ids(sim_matrix, drug_id):
    return sorted(range(len(sim_matrix)), key=lambda x: sim_matrix[drug_id, x], reverse=True)

#Filter the thetas so that they only mention the drugs and the pathways in the reference dataset
def filter_thetas(thetas, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names):
    filtered = np.delete(thetas, [i for i, p in enumerate(pathway_ids) if p not in pathways_in_eval], axis=1)
    filtered /= np.sum(filtered, axis=1)[:, np.newaxis]
    
    filtered = np.delete(filtered, [i for i, d in enumerate(drug_names) if d not in drug_names_in_eval], axis=0)
    
    return filtered
    
def calc_ref_ranks(thetas):
    return [np.where(d > 0)[0] for i, d in enumerate(thetas)]
    
def calculate_implied_theta(shape, eval_data):
    eval_thetas = np.zeros(shape)
    
    for i, e in enumerate(eval_data):
        for p in e:
            eval_thetas[i, p] = 1
            
    return eval_thetas / np.sum(eval_thetas, axis=1)[:, np.newaxis]

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
    
class ToyDataset():
    def __init__(self, doc_words, doc_counts, doc_thetas, mu, sigma, beta, m_params, thetas):
        self.doc_words = doc_words
        self.doc_counts = doc_counts
        self.doc_thetas = doc_thetas
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.m_params = m_params
        self.thetas = thetas

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
    
    return ToyDataset(doc_words, doc_counts, doc_thetas, mu, sigma, beta, m_params, thetas)

def save_toy_dataset(i, doc_words, doc_counts, doc_thetas, mu, sigma, beta, m_params, thetas):
    np.savez_compressed(diss_data_root + "%d-dataset" % i, doc_words, doc_counts, doc_thetas, mu, sigma, beta)
    np.savez_compressed(diss_data_root + "%d-results" % i, m_params, thetas)

def load_pathway_names():
    pathway_names = {}
    
    f = open(diss_data_root + "/pathway_names.txt", 'r')
    
    for name in list(f):
        name = name.replace('path:hsa', '').replace(' - Homo sapiens (human)\n', '').split('\t')
        pathway_names[int(name[0])] = name[0] + " " + name[1]
    
    f.close()
    return pathway_names

def atc(thetas):
    atc = pd.read_csv(diss_data_root + "ATC_codes_and_drug_names.csv")
    atc_dict = atc.set_index('DrugName')['ATC code'].to_dict()
    
    drug_names_in_atc = [d for d in drug_names if d in atc_dict]
    drug_labels = [atc_dict[d] for d in drug_names_in_atc]
    filtered_thetas = np.array([t for t, d in zip(thetas, drug_names) if d in drug_names_in_atc])
    
    #Turn the letter labels into indices
    
    sorted_labels = sorted(set(drug_labels))
    label_map = {l: i for i, l in enumerate(sorted_labels)}
    
    drug_labels = np.array([label_map[l] for l in drug_labels])

    #Create the classifier (100 trees, 8 threads) and cross-validate
    forest = RandomForestClassifier(n_estimators=100, n_jobs=8)
    samples = cross_validation.cross_val_score(forest, filtered_thetas, drug_labels, cv=10)
    print "%.3f, %.3f" % (np.mean(samples), np.std(samples))
    
    #nb: gmrf gets 0.172, 0.029
    
    
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

    drug_gene = np.loadtxt(diss_data_root + "gene_expression_matrix_X.txt.prev")
    drug_gene = np.exp(drug_gene)
    drug_gene = drug_gene[::drug_prune,::gene_prune]     
    
    doc_words = [d.nonzero()[0] for d in drug_gene]
    doc_counts = [d[d.nonzero()[0]] for d in drug_gene]

    priors = np.loadtxt(diss_data_root + "gene_pathway_matrix_K.txt.prev")[::pathway_prune,::gene_prune]
    priors = np.array([p / sum(p) for p in priors])
    beta_density = density(priors) #0.0138 for ctd

    print "Drugs: %d, pathways: %d, genes: %d" % (drug_gene.shape[0], priors.shape[0], drug_gene.shape[1])
### 
##    m_params, v_params = expectation_maximization(doc_words, doc_counts, len(priors), priors, max_iterations=100)
#    
###    
####    
    data = np.load(diss_data_root + "../data-every-1th-drug.npz")
    m_params = data['arr_0'].item()
###    v_params = data['arr_1']
###    
    data = np.load(diss_data_root + "../thetas-every-1th-drug.npz")
    thetas = data['arr_0']
    
#    data = np.load(diss_data_root + "../thetas-every-1th-drug-exp.npz")
#    thetas_exp = data['arr_0']

#   thetas = np.array([expected_theta(v, m_params, w, c) for v, w, c in zip(v_params, doc_words, doc_counts)])      

    pathway_ids = [int(p.strip()) for p in open(diss_data_root + "pathway_id.txt.prev").readlines()][::pathway_prune]
#    pathway_names = [p.strip() for p in open(diss_data_root + "pathway_names_used.txt").readlines()][::pathway_prune]
    drug_names = [d.strip() for d in open(diss_data_root + "drug_name.txt").readlines()][1::drug_prune]
    
    eval_data = load_evaluation_dataset(set(pathway_ids) - {1100}) 
    pathway_map = {v: k for k, v in enumerate(pathway_ids)}
    pathways_in_eval = set.union(*([set(eval_data[d]) for d in drug_names if d in eval_data]))
    pathway_ids_in_eval = sorted(pathways_in_eval)
    pathway_map_in_eval = {v: k for k, v in enumerate(pathway_ids_in_eval)}
#    pathway_names_in_eval = [n for i, n in zip(pathway_ids, pathway_names) if i in pathway_ids_in_eval]
    drug_names_in_eval = [d for d in drug_names if d in eval_data]
    eval_data = [[pathway_map_in_eval[p] for p in eval_data[d]] for d in drug_names_in_eval]

#Load the LDA phi values and transpose to have the same shape as our beta
#    lda_phi = np.loadtxt(diss_data_root + "lda-phi-paper.txt").T
#    lda_phi /= np.sum(lda_phi, axis=1)[:, np.newaxis]
#
#    lda_thetas = np.loadtxt(diss_data_root + "lda-theta-paper.txt")
#    lda_thetas_f = filter_thetas(lda_thetas, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names)
    
    ctm_thetas_f = filter_thetas(thetas, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names)
#    ctm_thetas_exp_f = filter_thetas(thetas_exp, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names)
    
    random_thetas = np.random.uniform(size=ctm_thetas_f.shape)
    random_thetas /= np.sum(random_thetas, axis=1)[:, np.newaxis]

#Construct thetas implied by the evaluation data
    eval_thetas = np.zeros(ctm_thetas_f.shape)
    
    for i, e in enumerate(eval_data):
        for p in e:
            eval_thetas[i, p] = 1
            
    eval_thetas /= np.sum(eval_thetas, axis=1)[:, np.newaxis]    
    theta_sparsity = density(eval_thetas) #0.1231 for ctd dataset

    
#    gmrf_thetas = filter_thetas(scipy.io.loadmat(diss_data_root + "output_s_me.mat")['output_S'].T, pathways_in_eval, pathway_ids, drug_names_in_eval, drug_names)
#    gmrf_thetas /= np.sum(gmrf_thetas, axis=1)[:, np.newaxis]

#    lda_perf = np.log(validate_all_thetas(lda_thetas_f, eval_data))
    ctm_perf = np.log(validate_all_thetas(ctm_thetas_f, eval_data))
#    ctm_exp_perf = validate_all_thetas(ctm_thetas_exp_f, eval_data)
    random_perf = validate_all_thetas(random_thetas, eval_data)
#    gmrf_perf = np.log(validate_all_thetas(gmrf_thetas, eval_data))

#    ctm_exp_side = np.sum(calc_heatmap(ctm_thetas_exp_f, eval_data), axis=1)
#    gmrf_side = np.sum(calc_heatmap(gmrf_thetas, eval_data), axis=1)
#    lda_side = np.sum(calc_heatmap(lda_thetas_f, eval_data), axis=1)
    ran_side = np.sum(calc_heatmap(random_thetas, eval_data), axis=1)
    ref_side = np.sum(calc_heatmap(eval_thetas, eval_data), axis=1)
    
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

    mu = np.loadtxt(diss_data_root + "ctd-implied-mu.txt")
    sigma = np.loadtxt(diss_data_root + "ctd-implied-sigma.txt")        
        
    voc_len = 3000
    K = len(sigma)
    N_d = 1000
    no_docs = 500
    
    print "Generating a random corpus..."
    doc_words, doc_counts, doc_thetas, mu, sigma, beta = generate_random_corpus(voc_len, K, N_d, no_docs, 0.1231, 0.0138, mu, sigma)
#    doc_words, doc_counts, doc_thetas, mu, sigma, beta = generate_random_corpus(voc_len, K, N_d, no_docs, 0.25, 0.0138)
    
#    validation(doc_words, doc_counts, doc_thetas)
    
    priors = np.ones(beta.shape)
    priors[beta == 0] = 0
    priors = np.array([p / sum(p) for p in priors])

    print "Performing expectation maximization..."    
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
    
#    #todo: axis=1 gives better results
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
    
    theta_diff_sizes = [dsm_rmse(inf, ref) for inf, ref in zip(permuted_thetas, doc_thetas)]

    ref_ranks = calc_ref_ranks(doc_thetas)
    
    thetas = np.array(thetas)
    doc_thetas = np.array(doc_thetas)
    
    baseline = generate_random_theta(doc_thetas)
    
    ctm_perf = validate_all_thetas(thetas, ref_ranks)
    rnd_perf = validate_all_thetas(baseline, ref_ranks)
    ref_perf = validate_all_thetas(doc_thetas, ref_ranks)
    
    ctm_side = np.sum(calc_heatmap(thetas, ref_ranks), axis=1)
    ref_side = np.sum(calc_heatmap(doc_thetas, ref_ranks), axis=1)
    rnd_side = np.sum(calc_heatmap(baseline, ref_ranks), axis=1)
    
    ctm_ap = [average_precision(t, r) for t, r in zip(thetas, ref_ranks)]
    rnd_ap = [average_precision(t, r) for t, r in zip(baseline, ref_ranks)]
    ref_ap = [average_precision(t, r) for t, r in zip(doc_thetas, ref_ranks)]
    
    ctm_cosines = [cosine_similarity(t, r) for t, r in zip(thetas, doc_thetas)]
    rnd_cosines = [cosine_similarity(t, r) for t, r in zip(baseline, doc_thetas)]

    reference = document_similarity_matrix(doc_thetas)
    inferred = document_similarity_matrix(thetas)
    
    print "RMSE between inferred document correlations and the reference: %f" % dsm_rmse(inferred, reference)
    print "RMSE between inferred beta and the reference: %f" % dsm_rmse(np.array(permuted_beta), np.array(beta))
    print "RMSE between inferred topic correlations and the reference: %f" % dsm_rmse(cor_mat(permuted_sigma), cor_mat(sigma))
    print "RMSE between inferred topic proportions and the reference: %f" % dsm_rmse(exp_normalise(permuted_mu), exp_normalise(mu))
    
        
    
#    