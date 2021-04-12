# --- IMPORTS --- #
from matplotlib.pylab import cm
from itertools import combinations, permutations, product
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

    Attributes:
        lhs (`tuple`): Itemset for the antecedent (left-hand side) of the rule.
        rhs (`tuple`): Itemset for the consequent (right-hand side) of the rule.
        support (`double`): Proportion of transactions where the items in the antecedent and consequent were present.
        confidence (`double`): Conditional probability that the consequent is present in the transaction given that the antecedent is present.
        lift (`double`): The proportionate rise in confidence that the presence of the antecedent in the transaction grants to the consequent.
    
    Methods:
        __str__: Returns a string representation of the rule.
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
cdef list product_names, clusters
cdef object graph, minimum_spanning_tree


cdef list __filter_rules(ruleset, min_support, min_confidence):
    """
    Filters out rules that do not satisfy specified parameters.

    Args:
        ruleset (`list`): List of tuples, each representing a rule's antecedent and consequent.
        min_support (`double`): Minimum support to qualify.
        min_confidence (`double`): Minimum confidence to qualify.
    
    Returns:
        rules (`list`): List of filtered rules.

    """
    is_above_threshold = True
    cdef list rules = []
    cdef set below_threshold = set()
    cdef double support_a, support_b, support_ab, confidence, lift
    cdef tuple a,b 

    for rule in ruleset:
        is_above_threshold = True
        a, b = rule
        a = tuple(sorted(a))
        b = tuple(sorted(b))
        ab = a+b

        for x in below_threshold:
            if set(x).issubset(ab):
                is_above_threshold = False
                break
        
        if is_above_threshold:
            support_a, support_b, support_ab = __support(a, b)
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
            
            rules.append(Rule(get_names(a), get_names(b), support_ab, confidence, lift))

    return rules


cdef list __add_reverse(list ruleset):
    """
    Appends the reversed tuple for each tuple in a ruleset.

    Args:
        ruleset (`list`): List of tuples, each representing the antecedent and consequent of a rule.
    
    Returns:
        rules (`list`): List of tuples, each representing the antecedent and consequent of a rule.

    """
    cdef list rules = []
    cdef tuple a,b
    for rule in ruleset:
        a, b = rule
        rules.append((a,b))
        rules.append((b,a))
    return rules


cpdef list generate_bicluster_rules(min_support=0.005, min_confidence=0.6):
    """
    Generates a list of bi-cluster rules, such that the antecedent and consequent of the rule
    originate from different clusters.

    Args:
        min_support (`double`): The minimum support required for the rule to be added (Default: 0.005).
        min_confidence (`double`): The minimum confidence required to rule to be added (Default: 0.6).
    
    Returns:
        rules (`list`): List of rules, each an instance of the `Rule` class.

    """
    cdef dict items_by_cluster = {}
    cdef set items
    for index, cluster in enumerate(clusters):
        items = set()
        for set_size in range(1, len(cluster)):
            items.update(combinations(cluster, set_size))
        items_by_cluster[index] = items
    
    cdef list rules = []
    cdef list cluster_combinations = list(combinations(list(range(len(clusters))), 2))

    for comb in cluster_combinations:
        a_items = items_by_cluster[comb[0]]
        b_items = items_by_cluster[comb[1]]
        current_rules = list(product(a_items, b_items))
        rules.extend(current_rules)
    
    rules = __add_reverse(rules)
    return __filter_rules(rules, min_support=min_support, min_confidence=min_confidence)

        

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
    Returns the number of possible rules for a ruleset of size 'd'

    """
    cdef int num = 0 # Number of rules so far

    for k in range(d):
        num += __comb(d, k) * sum([__comb(d-k, j) for j in range(d-k+1)])
    
    return num


def __support(a, b):
    """"
    Returns the proportion of transactions where the itemset was present, 
    for `a`, `b` and `a+b`.

    Args:
        a (`tuple`): Itemset.
        b (`tuple`): Itemset.
    
    Returns:
        a_filter (`double`): The proportion of transactions where items in `a` were present.
        b_filter (`double`): The proportion of transactions where items in `b` were present.
        ab_filter (`double`): The proportion of transactions where items in `a` and `b` were present.

    """"
    def get_condition(x):
        """
        Returns a conditional statement for `pandas` to filter the 
        dataset such that the items present in `x` are present in all 
        remaining transactions after filtration.

        Args:
            x (`tuple`): Itemset.
        
        Returns:
            condition (`object`): A conditional filter for `pandas`.

        """
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
    """
    Generates potential association rules from items within a specified cluster.

    Args:
        cluster (`tuple`): An itemset such that all children belong to a single cluster.
    
    Returns:
        pruned_rules (`list`): A list of tuples, such that each tuple represents the antecedent and consequent of a rule.

    """
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



cpdef init(filepath, pickled=False, graph_exists=False, inflation=None):
    """
    Initializes the state for the algorithm. Reads in the dataset, then generates 
    the graph and minimum spanning tree, and finally generates clusters.

    Args:
        filepath (`str`): Filepath for the dataset.
        pickled (`bool`): `True` if filepath is `pickle` format, `False` otherwise (Default: `False`).
        graph_exists (`bool`): `True` if the graph and MST have been saved as a pickle, `False` otherwise (Default: `False`).
        inflation (`double`): The inflation value to pass to the Markov Clustering algorithm (Default: None).

    """
    global df, product_names, graph, minimum_spanning_tree
    graph_path = f'data/output/graphs/{filepath.replace("/", "_")}_graph.pkl'
    mst_path = f'data/output/graphs/{filepath.replace("/", "_")}_mst.pkl'
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
        
    
    __generate_clusters(inflation=inflation)



cdef get_names(itemset):
    """
    Returns a tuple with the names associated with the items in an itemset.
    """
    return tuple(product_names[item] for item in itemset)

cpdef generate_rules(double min_support=0.005, double min_confidence=0.6):
    """
    Generates the association rules from within all clusters.

    Args:
        min_support (`double`): The minimum support required for a rule.
        min_confidence (`double`): The minimum confidence required for a rule.
    
    Returns:
        rules (`list`): A list of association rules, where each item is an instance of the `Rule` class.
    """
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


