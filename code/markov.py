# --- Imports --- #
from itertools import combinations
import markov_clustering as mc
import networkx as nx
import pandas as pd
import numpy as np
import time
import math


# ----- GLOBAL VARIABLES ----- #
df = None        # DataFrame with purchase vectors
corr = None      # Correlation matrix used for graph generation
columns = None   # Column names for products
G = None         # Graph
MST = None       # Minimum Spanning Tree from G
clusters = None  # Clusters from MST

def __distance_function(x):
    return np.sqrt(2*(1-x))


def __init():
    global df, corr, columns
    # Read in vectors
    df = pd.read_csv('data/rust_vectors.csv')

    # Generate Pearson's correlation matrix and adjust for distance function
    corr = np.array(__distance_function(df.corr()))

    # Assign columns 
    columns = df.columns
    
    # Assert no null values exist
    assert(not df.isnull().values.any())


def __get_graph():
    global G
    G = nx.from_numpy_matrix(corr)


def __get_mst():
    global MST
    MST = nx.minimum_spanning_tree(G)


def __get_optimal_clusters():
    global clusters
    matrix = nx.to_scipy_sparse_matrix(MST)
    best_score = (-math.inf)
    best_clusters = None
    inflation_ranges = np.arange(1.5, 2.6, 0.1)
    for inflation in inflation_ranges:
        result = mc.run_mcl(matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        Q = mc.modularity(matrix=result, clusters=clusters)
        if Q > best_score:
            best_score = Q
            best_clusters = clusters
    # Extract sparse matrix from MST
    clusters = best_clusters


def run():
    __init()
    __get_graph()
    __get_mst()
    __get_optimal_clusters()


def support(a):
    # Map to transaction
    transaction = (df.iloc[:, a[0]] == 1)

    for i in a[1:]:
        transaction = (transaction) & (df.iloc[:, i] == 1)

    condition = df[transaction]
    #print(f'{", ".join([columns[i] for i in item_indexes])}')
    #print(f'{condition.shape[0]:,}/{df.shape[0]:,} == {condition.shape[0]/df.shape[0]:,.2f}\n')
    return condition.shape[0]/df.shape[0]


def confidence(a, b):
    supp_a = support(a)
    supp_ab = support(a+b)
    return supp_ab/supp_a


def print_clusters():
    for x, cluster in enumerate(clusters):
        names = [f'{columns[i]} [{i}]' for i in cluster]
        print(f'Cluster #{x}: {", ".join(names)}\n')


def rules_within_clusters(min_support=0.0025, min_confidence=0.25):
    start = time.time()
    rules = []
    exclude = []
    ignored = 0

    # Lambda function returns true if no element from a exists in b
    is_unique = lambda a, b: len(set(a+b)) == (len(a)+len(b))
    # For each cluster
    for x, cluster in enumerate(clusters):
        # Set sizes
        for set_size in range(2, int(len(cluster)/2), 1):
            potential_rules = combinations(combinations(cluster, set_size), 2)
            for r in potential_rules:
                # unpack
                a, b = r
                ab = set(a+b)
                if not is_unique(a, b):
                    continue

                # Ignore if in exclusion list
                exclusion_matches = [ab.issubset(x) for x in exclude]
                if any(exclusion_matches):
                    ignored += 1
                    continue

                supp_val = support(a+b)
                
                # If below threshold, add to exclude and move to next
                if supp_val < min_support:
                    exclude.append(a+b)
                    ignored += 1
                    continue
                
                confidence_val = confidence(a, b)
                # If threshold met, add to rules
                rules.append([a,b, supp_val, confidence_val])
    
    print(f'Ignored {ignored:,} rules.')
    return rules




    
if __name__ == '__main__':
    run()

    start = time.time()
    rules = rules_within_clusters()
    end = time.time() - start
    print(f'It took {end:,.2f} seconds for rule generation.')
    # Sort by confidence
    rules.sort(key=lambda x: x[2], reverse=True)

    print(f'There are {len(rules):,} rules, showing top 15 by support.')
    for rule in rules[:15]:
        a = [columns[i] for i in rule[0]]
        b = [columns[i] for i in rule[1]]
        print(f'({", ".join(a)}) -> ({", ".join(b)})  |  Support: {rule[2]:.5f}   |  Confidence: {rule[3]:.5f}')