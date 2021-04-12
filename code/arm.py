# --- Imports --- #
from matplotlib.pylab import cm
from itertools import combinations, permutations
import matplotlib.pyplot as plt
import markov_clustering as mc
import networkx as nx
import pandas as pd
import numpy as np
import time
import math
import csv

# --- DEBUG FLAG --- #
DEBUG_MODE = False


def log(message):
    if DEBUG_MODE:
        print(message)


class Rule:
    def __init__(self, lhs, rhs, support, confidence, lift):
        self.lhs = lhs
        self.rhs = rhs
        self.support = support
        self.confidence = confidence
        self.lift = lift
    
    def __str__(self):
        return f'{", ".join(self.lhs)} -> {", ".join(self.rhs)}'
    

class MSTARM:
    """
    TODO: describe
    """

    # ------ PRIVATE FUNCTIONS ------ #
    def __log(self, message) -> None:
        if self.DEBUG_MODE:
            print(message)


    def __distance_function(self, x) -> float:
        """
        Converts the correlation matrix values such
        that the larger the value, the smaller the transformed
        value.

        Args:
            x (`pandas.DataFrame`): The correlation matrix.
        """
        return np.sqrt(2*(1-x))
    

    def __generate_clusters(self) -> list:
        """
        """
        matrix = nx.to_scipy_sparse_matrix(self.MST)
        best_score = -math.inf
        best_clusters = None
        inflation_ranges = np.arange(1.5, 2.6, 0.1)

        for val in inflation_ranges:
            result = mc.run_mcl(matrix, inflation=val)
            clusters = mc.get_clusters(result)
            Q = mc.modularity(matrix=result, clusters=clusters)
            if Q > best_score:
                best_score = Q
                best_clusters = clusters
        self.clusters = best_clusters


    def __support(self, a, b):
        
        def get_condition(x):
            # Checks if corresponding column for each element is 1
            condition = (self.df.iloc[:, x[0]] == 1)
            for i in x[1:]:
                condition = (condition) & (self.df.iloc[:, i] == 1)
            return condition

        
        a_filter = self.df[get_condition(a)].shape[0]/self.df.shape[0]
        b_filter = self.df[get_condition(b)].shape[0]/self.df.shape[0]
        ab_filter = self.df[get_condition(a+b)].shape[0]/self.df.shape[0]
        
        return a_filter, b_filter, ab_filter

    
    def __get_rules(self, cluster) -> list:
        rules = set()
        for set_size in range(1, len(cluster)):
            rules.update(combinations(cluster, set_size))
        rules = list(combinations(rules, 2))

        pruned_rules = []
        for rule in rules:
            a,b = rule
            if any(p in b for p in a):
                continue
            pruned_rules.append((a,b))
            pruned_rules.append((b,a))
        pruned_rules.sort(key=lambda x : len(x[0]))
        return pruned_rules

    
    

    
    
    def __init__(self, filepath, debug_mode=True, exclude_one_to_one=False) -> None:
        self.exclude_one_to_one = exclude_one_to_one
        self.DEBUG_MODE = debug_mode
        self.df = pd.read_csv(filepath)
        self.names = self.df.columns
        self.corr = np.array(self.__distance_function(self.df.corr()))
        self.G = nx.from_numpy_matrix(self.corr)
        self.MST = nx.minimum_spanning_tree(self.G)
        self.__generate_clusters()

    

    # ------ PUBLIC FUNCTIONS ------ #

    def generate_bicluster_rules(self, min_support=0.005, min_confidence=0.6):
        items_by_cluster = {}
        for index, cluster in enumerate(self.clusters):
            items = set()
            for set_size in range(1, len(cluster)):
                items.update(combinations(cluster, set_size))
            items_by_cluster[index] = items
        
        rules = []

        cluster_combinations = list(combinations(list(range(len(self.clusters))), 2))

        for comb in cluster_combinations:
            a_items = items_by_cluster[comb[0]]
            b_items = items_by_cluster[comb[1]]
            
            current_rules = [list(zip(i,b_items)) for i in permutations(a_items,len(b_items))]
            rules.extend(current_rules)
        
        pruned_rules = []
        pruned_count = 0
        for rule in rules:
            a, b = rule
            if any(p in b for p in a):
                pruned_count += 1
                continue
            pruned_rules.append((a,b))
            pruned_rules.append((b,a))
        
        print(f'Removed {pruned_count:,} duplicates.')
        
        below_threshold = set()
        final_rules = []
        for rule in pruned_rules:
            is_above_threshold = True
            a,b = rule
            a = tuple(sorted(a))
            b = tuple(sorted(b))
            ab = a + b

            for x in below_threshold:
                if set(x).issubset(ab):
                    is_above_threshold = False
                    break
                    
            if is_above_threshold:
                support_a, support_b, support_ab = self.__support(a,b)
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

                a_str = tuple(self.names[i] for i in a)
                b_str = tuple(self.names[i] for i in b)

                final_rules.append(Rule(a_str, b_str, support_ab, confidence, lift))
        
        return final_rules


        



    def plot_graph_and_mst(self, dim=10, output_filepath=None, layout_on_mst=False):
        """
        TODO: Document
        """

        # Plot layout
        nrows, ncols = 1,2
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(dim*ncols, dim*nrows))

        # Set Titles
        ax[0].set_title("Complete Graph", fontsize=2*dim)
        ax[1].set_title("Minimum Spanning Tree", fontsize=2*dim)

        # Fixed positional layout for nodes
        pos = nx.spring_layout(self.MST) if layout_on_mst else nx.spring_layout(self.G)

        # Collection of plotting arguments common amongst the two draw functions
        kwargs = dict(pos=pos, with_labels=True, node_color='c', node_size=400, edge_color='0.25', font_size=1.5*dim)

        # Plot full graph
        nx.draw(self.G, ax=ax[0], width=1, **kwargs)
        # Plot MST
        nx.draw(self.MST, ax=ax[1], width=2, **kwargs)

        if output_filepath is not None:
            plt.savefig(output_filepath)
        
        plt.show()

    
    def plot_mst_clusters(self, dim=10, output_filepath=None):
        """
        TODO: Document
        """

        # Plot layout
        nrows, ncols = 1,2
        _, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(dim*ncols, dim*nrows))

        # Set titles
        ax[0].set_title("Minimum Spanning Tree", fontsize=20)
        ax[1].set_title("Markov Clustered MST", fontsize=20)

        # Fixed positional layout for nodes
        pos = nx.spring_layout(self.MST)

        # Collection of plotting arguments common amongst the two draw functions
        kwargs= dict(pos=pos, with_labels=True,node_size=400, edge_color='0.25', font_size=15)

        # map node to cluster id for colors
        cluster_map = {node: i for i, cluster in enumerate(self.clusters) for node in cluster}
        colors = [cluster_map[i] for i in range(len(self.MST.nodes()))]

        # Plot unclustered MST
        nx.draw(self.MST, ax=ax[0], node_color='c', width=1, **kwargs)
        # Plot clustered MST
        nx.draw(self.MST, node_color=colors, cmap=cm.tab20, ax=ax[1], width=2, **kwargs)

        if output_filepath is not None:
            plt.savefig(output_filepath)


    def generate_rules(self, min_support=0.005, min_confidence=0.6) -> None:
        # Check if clusters initialized
        assert(self.clusters is not None)
        # Record rules below threshold for pre-emptive pruning
        below_threshold = set()
        # Record rules to return
        self.rules = []

        # -- TESTING VARIABLES -- #
        below_threshold_count = 0


        for cluster_num, cluster in enumerate(self.clusters):
            cluster_rules = self.__get_rules(cluster)
            # sort by size of antecedent, allowing the pruning of larger sets later.

            
            for rule in cluster_rules:
                is_above_threshold = True # Default, will change if found to be below threshhold subset
                a, b = rule
                # sort
                a = tuple(sorted(a))
                b = tuple(sorted(b))
                ab = a + b
                
                # Flag if rule is a subset of sets that are below the threshold
                for x in below_threshold:
                    if set(x).issubset(ab):
                        below_threshold_count += 1
                        is_above_threshold = False
                        break

                # If rule does not match any that are below threshold
                if is_above_threshold:
                    # Used to avoid re-calculation of same values
                    support_a, support_b, support_ab = self.__support(a, b)
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

                    a_str = tuple(self.names[i] for i in a)
                    b_str = tuple(self.names[i] for i in b)
                    
                    new_rule = Rule(a_str, b_str, support_ab, confidence, lift)
                    self.rules.append(new_rule)
        return self.rules

    
    def identify_cluster(self, product):
        for i, cluster in enumerate(self.clusters):
            if product in [self.names[q] for q in cluster]:
                return i




if __name__ == '__main__':
    start = time.time()

    # ----- Write Code Here ----- #
    arm = MSTARM('data/rust_vectors_product_category_no_fuel.csv')
    arm.generate_rules()
    # ----- End Code Here ----- #
    end = time.time() - start
    print(f'Finished in {end:,.2f} seconds.')

