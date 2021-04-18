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
    """
    Class to represent an association rule.
    Attributes:
        lhs (`tuple`): Itemset for the antecedent (left-hand side) of the rule.
        rhs (`tuple`): Itemset for the consequent (right-hand side) of the rule.
        support (`float`): Proportion of transactions where the items in the antecedent and consequent were present.
        confidence (`float`): Conditional probability that the consequent is present in the transaction given that the antecedent is present.
        lift (`float`): The proportionate rise in confidence that the presence of the antecedent in the transaction grants to the consequent.
    
    Methods:
        __str__: Returns a string representation of the rule.
    """
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
    """
    Filters out rules that do not satisfy specified parameters.
    Args:
        ruleset (`list`): List of tuples, each representing a rule's antecedent and consequent.
        min_support (`float`): Minimum support to qualify.
        min_confidence (`float`): Minimum confidence to qualify.
    
    Returns:
        rules (`list`): List of filtered rules.
    """
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
    """
    Appends the reversed tuple for each tuple in a ruleset.
    Args:
        ruleset (`list`): List of tuples, each representing the antecedent and consequent of a rule.
    
    Returns:
        rules (`list`): List of tuples, each representing the antecedent and consequent of a rule.
    """
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
    """
    Returns the proportion of transactions where the itemset 'x' was present.
    Args:
        x (`tuple`): itemset.
    
    Returns:
        (`float`): The support score.
    """
    condition = (df.iloc[:, x[0]] == 1)
    for i in x[1:]:
        condition = (condition) & (df.iloc[:, i] == 1)
    
    return df[condition].shape[0]/df.shape[0]


def __prune_rules(ruleset):
    """
    Prunes rules so that there is no intersection between the antecedent and consequent.
    Args:
        ruleset (`list`): Ruleset.
    
    Returns:
        pruned (`list`): Ruleset.
    """
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
    """
    Initializes the state for the algorithm. Reads in the dataset, then generates 
    the graph and minimum spanning tree, and finally generates clusters.
    Args:
        filepath (`str`): Filepath for the dataset.
        pickled (`bool`): `True` if filepath is `pickle` format, `False` otherwise (Default: `False`).
        graph_exists (`bool`): `True` if the graph and MST have been saved as a pickle, `False` otherwise (Default: `False`).
        inflation (`float`): The inflation value to pass to the Markov Clustering algorithm (Default: None).
    """
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
    """
    Generates a list of bi-cluster rules, such that the antecedent and consequent of the rule
    originate from different clusters.
    Args:
        min_support (`double`): The minimum support required for the rule to be added (Default: 0.005).
        min_confidence (`double`): The minimum confidence required to rule to be added (Default: 0.6).
    
    Returns:
        rules (`list`): List of rules, each an instance of the `Rule` class.
    """
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

def plot_graph_and_mst(dim=10, output_filepath=None, layout_on_mst=False):
    """
    Plots the complete graph and minimum spanning tree.
    Args:
        dim (`int`): The dimension for each block in the graph (Default: 10).
        output_filepath (`str`): The filepath to save the plot image to, if specified (Default: None).
        layout_on_mst (`bool`): Sets the layout based on the MST if `True`, on the graph otherwise (Default: `False`).
    """
    # Plot layout
    nrows, ncols = 1,2
    _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(dim*ncols, dim*nrows))

    # Set Titles
    ax[0].set_title("Complete Graph", fontsize=2*dim)
    ax[1].set_title("Minimum Spanning Tree", fontsize=2*dim)

    # Fixed positional layout for nodes
    pos = nx.spring_layout(minimum_spanning_tree) if layout_on_mst else nx.spring_layout(graph)

    # Collection of plotting arguments common amongst the two draw functions
    kwargs = dict(pos=pos, with_labels=True, node_color='c', node_size=400, edge_color='0.25', font_size=1.5*dim)

    # Plot full graph
    nx.draw(graph, ax=ax[0], width=1, **kwargs)
    # Plot MST
    nx.draw(minimum_spanning_tree, ax=ax[1], width=2, **kwargs)

    if output_filepath is not None:
        plt.savefig(output_filepath)
    
    plt.show()


def plot_mst_clusters(dim=10, output_filepath=None):
        """
        Plots the clustered and unclustered MSTs.
        Args:
            dim (`int`): The dimension for each block in the graph (Default: 10).
            output_filepath (`str`): The filepath to save the plot image to, if specified (Default: None).
        
        """
        # Plot layout
        nrows, ncols = 1,2
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(dim*ncols, dim*nrows))

        # Set titles
        ax[0].set_title("Minimum Spanning Tree", fontsize=20)
        ax[1].set_title("Markov Clustered MST", fontsize=20)

        # Fixed positional layout for nodes
        pos = nx.spring_layout(minimum_spanning_tree)

        # Collection of plotting arguments common amongst the two draw functions
        kwargs= dict(pos=pos, with_labels=True,node_size=400, edge_color='0.25', font_size=15)

        # map node to cluster id for colors
        cluster_map = {node: i for i, cluster in enumerate(clusters) for node in cluster}
        colors = [cluster_map[i] for i in range(len(minimum_spanning_tree.nodes()))]

        # Plot unclustered MST
        nx.draw(minimum_spanning_tree, ax=ax[0], node_color='c', width=1, **kwargs)
        # Plot clustered MST
        nx.draw(minimum_spanning_tree, node_color=colors, cmap=cm.tab20, ax=ax[1], width=2, **kwargs)

        if output_filepath is not None:
            plt.savefig(output_filepath)
        
        plt.show()