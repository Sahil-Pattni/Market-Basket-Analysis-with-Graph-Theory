# --- Imports --- #
from itertools import combinations

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


    def __support(self, a, b=None) -> float:
        if b is not None:
            a += b
        
        # Add condition for binary validation of purchases.
        condition = (self.df.iloc[:, a[0]] == 1)
        for i in a[1:]:
            condition = (condition) & (self.df.iloc[:, i] == 1)
        
        transaction = self.df[condition]
        return transaction.shape[0]/self.df.shape[0]

    
    def __get_rules(self, cluster) -> list:
        # Excludes 1-1 rules
        def exclude(rule):
            a,b = rule
            if len(a) == len(b):
                if len(a) == 1:
                    return False
            return True

        rules = set()
        for set_size in range(1, len(cluster)):
            rules.update(combinations(cluster, set_size))
        rules = list(combinations(rules, 2))
        # Filter out one to one rules if specified
        if self.exclude_one_to_one:
            rules = list(filter(exclude, rules))

        return rules
    
    
    def __init__(self, filepath, debug_mode=False, exclude_one_to_one=False) -> None:
        self.exclude_one_to_one = exclude_one_to_one
        self.DEBUG_MODE = debug_mode
        self.df = pd.read_csv(filepath)
        self.names = self.df.columns
        self.corr = np.array(self.__distance_function(self.df.corr()))
        self.G = nx.from_numpy_matrix(self.corr)
        self.MST = nx.minimum_spanning_tree(self.G)
        self.__generate_clusters()
        self.__log('Successfully initialized instance with graphs.')
        self.scanned_rules = set()
    

    # ------ PUBLIC FUNCTION ------ #
    def generate_rules(self, min_support=0.005, min_confidence=0.6) -> None:
        # Check if clusters initialized
        assert(self.clusters is not None)
        start_all = time.time()
        # Record rules below threshold for pre-emptive pruning
        below_threshold = set()
        # Record rules to return
        self.rules = []

        # -- TESTING VARIABLES -- #
        already_scanned = 0
        not_unique = 0
        below_threshold_count = 0

        support_times = []


        for cluster in self.clusters:
            cluster_rules = self.__get_rules(cluster)
            # sort by size of antecedent, allowing the pruning of larger sets later.
            cluster_rules.sort(key=lambda x: len(x[0]))

            self.__log(f'Running {len(cluster_rules):,} rules.')
            
            for rule in cluster_rules:
                a, b = rule
                # sort
                a = tuple(sorted(a))
                b = tuple(sorted(b))
                ab = a + b
                
                if any([set(x).issubset(ab) for x in below_threshold]):
                    below_threshold_count += 1
                    continue

                # Used to avoid re-calculation of same values
                start = time.time()
                support_a = self.__support(a)
                support_times.append(time.time()-start)
                start = time.time()
                support_b = self.__support(b)
                support_times.append(time.time()-start)
                start = time.time()
                support_ab = self.__support(a, b)
                support_times.append(time.time()-start)
                confidence = 0 if support_a == 0 else support_ab/support_a
                lift = 0 if support_b == 0 else confidence/support_b

                if support_ab < min_support:
                    below_threshold.add(ab)
                    continue
                
                if confidence < min_confidence:
                    continue

                a_str = tuple(self.names[i] for i in a)
                b_str = tuple(self.names[i] for i in b)
                
                new_rule = Rule(a_str, b_str, support_ab, confidence, lift)
                self.rules.append(new_rule)
        
        print(f'Finished generating rules in {(time.time() - start_all):,.2f} seconds')
        print(f'Already Scanned: {already_scanned:,}\nNot Unique: {not_unique:,}\nBelow Threshold: {below_threshold_count:,}')
        print(f'Average time to calculate support: {sum(support_times)/len(support_times):.5f} for {len(support_times):,} iterations.')


        return self.rules

    
    def identify_cluster(self, product):
        for i, cluster in enumerate(self.clusters):
            if product in [self.names[q] for q in cluster]:
                return i

if __name__ == '__main__':
    arm = MSTARM('data/rust_vectors.csv', debug_mode=True)
    arm.generate_rules()



    
