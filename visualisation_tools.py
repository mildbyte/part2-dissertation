# -*- coding: utf-8 -*-
"""
Created on Mon Apr 06 21:36:46 2015

@author: mildbyte
"""

import pygraphviz
import pyplot as plt
import numpy as np

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

def generate_cluster_graph(mcl_result, labels=None):
    K = mcl_result.shape[0]
    if labels is None:
        labels = xrange(K)
        
    G = pygraphviz.AGraph('graph G {}')
    G.node_attr['shape'] = 'box'
    G.graph_attr['splines'] = 'spline'
    G.graph_attr['overlap'] = 'prism'
    
    for a in xrange(K):
        for b in xrange(K):
            if a != b and mcl_result[a, b] > 1e-5:
                G.add_edge(labels[a], labels[b])
    
    degree = G.degree()
    G.delete_nodes_from([n for i, n in enumerate(G.nodes()) if degree[i] == 0])

    return G
    
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
    G.delete_nodes_from([n for i, n in enumerate (G.nodes()) if degree[n] == 0])

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
    

def calc_heatmap(thetas, eval_data):
    
    ranks = [sorted(range(len(t)), key=lambda x: t[x], reverse=True) for t in thetas]
    img = np.zeros(thetas.T.shape)
    
    for d, reference, rank in zip(xrange(len(eval_data)), eval_data, ranks):
        for i, pathway in enumerate(rank):
            img[i, d] = 1 if pathway in reference else 0
    
    return img
    
def plot_heatmap(thetas, eval_data):
    imshow(1 - calc_heatmap(thetas, eval_data), cmap="Greys_r", interpolation='nearest')

#Performs Markov clustering on the similarity matrix
def mcl(M, power, inflation):
    while True:
        Mprev = M
        M = M / np.sum(M, axis=0)
        
        M = np.linalg.matrix_power(M, inflation)
        M = M ** power
        
        if (np.mean(np.abs(M - Mprev)) < 0.00001):
            return M / np.sum(M, axis=0)

def mcl_it(M, power, inflation, iterations=100):
    for _ in xrange(iterations):
        M = M / np.sum(M, axis=0)
        
        M = M ** power
        M = np.linalg.matrix_power(M, inflation)
        
    return M / np.sum(M, axis=0)