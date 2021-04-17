# --- IMPORTS --- #
from matplotlib.pylab import cm
from itertools import combinations, permutations, product
from functools import lru_cache
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

class Rule:
    def __init__(self, lhs: tuple, rhs: tuple, support: float, confidence: float, lift: float):
        self.lhs = lhs
        self.rhs = rhs
        self.support = support
        self.confidence = confidence
        self.lift = lift
    
    def __str__(self):
        return f'{self.lhs} -> {self.rhs}\nsupport: {self.support:.4f} | confidence: {self.confidence:.4f} | lift {self.lift:.4f}'


# GLOBALS
df = None
product_names, clusters = None, None
graph, minimum_spanning_tree = None, None
itemsets_by_cluster = {}


def __get_names(itemset):
    """
    Returns a tuple with the names associated with the items in an itemset.
    """
    return tuple(product_names[item] for item in itemset)


def __filter_rules(ruleset, min_support, min_confidence):
    rules = []
    below_threshold = set()

    for rule in ruleset:
        is_above_threshold = True
        a, b = rule
        a = tuple(sorted(a))
        b = tuple(sorted(b))
        ab = a + b

        for x in below_threshold:
            if set(x).issubset(ab):
                is_above_threshold = False
                break
        
        if is_above_threshold:
            # support_a, support_b, support_ab = __support(a, b)
            support_a = __support(a)
            support_b = __support(b)
            support_ab = __support(ab)
            confidence = 0 if support_a == 0 else support_ab/support_a
            lift = 0 if support_b == 0 else confidence/support_b

            if support_a < min_support:
                below_threshold.add(a)
            
            if support_b < min_support:
                below_threshold.add(b)
            
            if confidence < min_confidence:
                continue

            if support_ab < min_support:
                continue
            
            rules.append(Rule(__get_names(a), __get_names(b), support_ab, confidence, lift))
    return rules


def __add_reverse(ruleset):
    rules = []
    for rule in ruleset:
        a, b = rule
        rules.append((a,b))
        rules.append((b,a))
    return rules

def __distance_function(x):
    """ Vectorized Numpy matrix transformation function. """
    return np.sqrt(2*(1-np.abs(x)))

@lru_cache
def __support(x):
    condition = (df.iloc[:, x[0]] == 1)
    for i in x[1:]:
        condition = (condition) & (df.iloc[:, i] == 1)
    
    return df[condition].shape[0]/df.shape[0]


def __prune_rules(ruleset):
    pruned = []
    for rule in ruleset:
        a, b = rule
        is_valid = True
        for item in a:
            if item in b:
                is_valid = False
                break
        if is_valid:
            pruned.append(rule)
    
    pruned.sort(key=lambda x: len(x[0]))
    return pruned

def __generate_clusters(inflation=None):
    """
    Clusters the minimum spanning tree using Markov Clustering using a specified 
    inflation value if provided, or after determining the optimal value otherwise.

    Args:
        inflation (`double`): The inflation value for the Markov Clustering Algorithm (Default: None).

    """
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

def __generate_itemsets_by_cluster():
    """ Generates all possible itemsets for each cluster. """
    global itemsets_by_cluster
    for index, cluster in enumerate(clusters):
        items = set()
        for set_size in range(1, len(cluster)):
            items.update(combinations(cluster, set_size))
        itemsets_by_cluster[index] = items

def init(filepath, pickled=False, graph_exists=False, inflation=None):
    global df, product_names, graph, minimum_spanning_tree
    graph_path = f'data/output/graphs/{filepath.replace("/", "_")[:-4]}_graph.pkl'
    mst_path = f'data/output/graphs/{filepath.replace("/", "_")[:-4]}_mst.pkl'

    df = pd.read_pickle(filepath) if pickled else pd.read_csv(filepath)
    product_names = list(df.columns)
    corr = __distance_function(np.array(df.corr()))

    if graph_exists:
        graph = nx.read_gpickle(graph_path)
        minimum_spanning_tree = nx.read_gpickle(mst_path)
    else:
        graph = nx.from_numpy_matrix(corr)
        minimum_spanning_tree = nx.minimum_spanning_tree(graph)
        nx.write_gpickle(graph, graph_path)
        nx.write_gpickle(minimum_spanning_tree, mst_path)
        
    
    __generate_clusters(inflation=inflation)
    __generate_itemsets_by_cluster()


def generate_bicluster_rules(min_support=0.005, min_confidence=0.6):
    rules = []
    cluster_combinations = list(combinations(list(range(len(clusters))), 2))

    for comb in cluster_combinations:
        a_items = itemsets_by_cluster[comb[0]]
        b_items = itemsets_by_cluster[comb[1]]
        current_rules = list(product(a_items, b_items))
        rules.extend(current_rules)
    
    rules = __add_reverse(rules) # More efficient than calling permutations() in `cluster_combinations` above.
    return __filter_rules(rules, min_support=min_support, min_confidence=min_confidence)


def generate_intracluster_rules(min_support=0.005, min_confidence=0.6):
    """
    Generates the association rules from within a cluster for each cluster.

    Args:
        min_support (`float`): The minimum support required for a rule.
        min_confidence (`float`): The minimum confidence required for a rule.
    
    Returns:
        rules (`list`): A list of association rules, where each item is an instance of the `Rule` class.
    """
    below_threshold = set()
    rules = []

    for _, itemsets in itemsets_by_cluster.items():
        ruleset = list(combinations(itemsets, 2))
        ruleset = __prune_rules(ruleset)
        ruleset = __add_reverse(ruleset)
        # TODO: Add reverse
        for rule in ruleset:
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
                support_a = __support(a)
                support_b = __support(b)
                support_ab = __support(ab)
                confidence = 0 if support_a == 0 else support_ab / support_a
                lift = 0 if support_b == 0 else confidence / support_b

                if support_a < min_support:
                    below_threshold.add(a)
                
                if support_b < min_support:
                    below_threshold.add(b)

                if confidence < min_confidence:
                    continue

                if support_ab < min_support:
                    continue
                
                rules.append(Rule(__get_names(a), __get_names(b), support_ab, confidence, lift))
    return rules