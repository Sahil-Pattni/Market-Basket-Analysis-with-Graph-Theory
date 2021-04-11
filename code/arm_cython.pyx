# --- IMPORTS --- #
from matplotlib.pylab import cm
from itertools import combinations
import matplotlib.pyplot as plt
import markov_clustering as mc
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import time
import math
import csv
# --- END IMPORTS --- #



cdef class Rule:
    """
    Class to represent an association rule.
    """
    cdef tuple lhs, rhs
    cdef double support, confidence, lift
    def __init__(self, lhs, rhs, support, confidence, lift):
        self.lhs = lhs
        self.rhs = rhs
        self.support = support
        self.confidence = confidence
        self.lift = lift
    
    def __str__(self):
        return f'{self.lhs} -> {self.rhs} | support: {self.support:.4f} | confidence: {self.confidence:.4f} | lift {self.lift:.4f}'


# --- GLOBAL VARIABLES --- #
cdef object df
cdef list rules, product_names, clusters, cluster_rules
cdef object graph, minimum_spanning_tree

def __distance_function(x):
    """ Vectorized Numpy matrix transformation function """
    return np.sqrt(2*(1-x))


cdef int __factorial(int n):
    """ Returns the factorial of a number """
    cdef int i, ret
    ret = 1

    for i in range(n):
        ret *= n
    return ret


cdef int __comb(int n, int k):
    """ Returns nCk (combination) """
    return __factorial(n)/(__factorial(k) * __factorial(n-k))


cdef int __num_rule(int d):
    """
    Returns the number of possible rules for a 
    ruleset of size 'd'
    """
    cdef int num = 0 # Number of rules so far

    for k in range(d):
        num += __comb(d, k) * sum([__comb(d-k, j) for j in range(d-k+1)])
    
    return num


def __support(a, b):      
    def get_condition(x):
        # Checks if corresponding column for each element is 1
        condition = (df.iloc[:, x[0]] == 1)
        for i in x[1:]:
            condition = (condition) & (df.iloc[:, i] == 1)
        return condition

    num_rows = df.shape[0]
    a_filter = df[get_condition(a)].shape[0]/num_rows
    b_filter = df[get_condition(b)].shape[0]/num_rows
    ab_filter = df[get_condition(a+b)].shape[0]/num_rows
    
    return a_filter, b_filter, ab_filter


def __get_rules(cluster):
    rules = []
    for set_size in range(1, len(cluster)):
        rules.extend(list(combinations(cluster, set_size)))
   
    rules = list(combinations(rules, 2))

    # Prune where elements are in both antecedent and consequent
    pruned_rules = []
    for rule in rules:
        a, b = rule
        if any(p in b for p in a):
            continue
        pruned_rules.append((a,b))
        pruned_rules.append((b,a))

    pruned_rules.sort(key=lambda x: len(x[0]))
    return pruned_rules


def __generate_clusters(inflation=None):
    global clusters
    matrix = nx.to_scipy_sparse_matrix(minimum_spanning_tree)

    if inflation is not None:
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
    else:
        best_score = -math.inf
        best_clusters = None
        for val in np.arange(1.5, 2.6, 0.1):
            result = mc.run_mcl(matrix, inflation=val)
            current_clusters = mc.get_clusters(result)
            modularity = mc.modularity(matrix=result, clusters=current_clusters)
            if modularity > best_score:
                best_score = modularity
                best_clusters = current_clusters
        clusters = best_clusters



cpdef init(filepath, pickled=True, graph_exists=False):
    global df, product_names, graph, minimum_spanning_tree
    graph_path = 'data/output/graphs/graph.pkl'
    mst_path = 'data/output/graphs/mst.pkl'
    df = pd.read_pickle(filepath) if pickled else pd.read_csv(filepath)
    product_names = list(df.columns)
    cdef object corr = __distance_function(np.array(df.corr()))

    if graph_exists:
        graph = nx.read_gpickle(graph_path)
        minimum_spanning_tree = nx.read_gpickle(mst_path)
    else:
        graph = nx.from_numpy_matrix(corr)
        minimum_spanning_tree = nx.minimum_spanning_tree(graph)
        nx.write_gpickle(graph, graph_path)
        nx.write_gpickle(minimum_spanning_tree, mst_path)
        
    
    __generate_clusters(inflation=1.6)



cdef get_names(itemset):
    return tuple(product_names[item] for item in itemset)

cpdef generate_rules(double min_support=0.005, double min_confidence=0.6):
    cdef set below_threshold = set()
    cdef list rules = []
    cdef tuple a, b
    cdef double support_a, support_b, support_ab, confidence, lift

    for cluster in clusters:
        cluster_rules = __get_rules(cluster)

        for rule in cluster_rules:
            is_above_threshold = True
            atup, btup = rule
            a = tuple(sorted(atup))
            b = tuple(sorted(btup))
            ab = a + b

            for x in below_threshold:
                if set(x).issubset(ab):
                    is_above_threshold = False
                    break
            
            if is_above_threshold:
                support_a, support_b, support_ab = __support(a, b)
                confidence = 0 if support_a == 0 else support_ab / support_a
                lift = 0 if support_b == 0 else confidence / support_b

                if support_a < min_support:
                    below_threshold.add(a)
                
                if support_b < min_support:
                    below_threshold.add(b)

                if confidence < min_confidence:
                    continue
                
                rules.append(Rule(get_names(a), get_names(b), support_ab, confidence, lift))
    return rules


