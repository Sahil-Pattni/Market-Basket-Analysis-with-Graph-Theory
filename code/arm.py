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
        
        condition = (self.df.iloc[:, a[0]] == 1)
        for i in a[1:]:
            condition = (condition) & (self.df.iloc[:, i] == 1)
        transaction = self.df[condition]
        return transaction.shape[0]/self.df.shape[0]

    
    def __get_rules(self, cluster) -> list:
        rules = set()
        for set_size in range(1, len(cluster)):
            rules.update(combinations(cluster, set_size))
        return list(combinations(rules, 2))

    
    def __write_rules_to_csv(self) -> None:
        assert(self.rules is not None)
        with open('data/output/class_rules.csv', 'w+') as f:
            writer = csv.writer(f)
            for row in self.rules:
                writer.writerow(row)
    
    
    def __init__(self, filepath, debug_mode=False) -> None:
        self.DEBUG_MODE = debug_mode
        self.df = pd.read_csv('data/rust_vectors.csv')
        self.names = self.df.columns
        self.corr = np.array(self.__distance_function(self.df.corr()))
        self.G = nx.from_numpy_matrix(self.corr)
        self.MST = nx.minimum_spanning_tree(self.G)
        self.__generate_clusters()
        self.__log('Successfully initialized instance with graphs.')
    

    # ------ PUBLIC FUNCTION ------ #
    def generate_rules(self, min_support=0.005, min_confidence=0.6, write_to_csv=False) -> None:
        assert(self.clusters is not None)
        start = time.time()
        below_threshold = set()
        is_unique = lambda a,b: len(set(a+b)) == (len(a) + len(b))
        excluded, ignored = 0,0
        self.rules = []

        for x, cluster in enumerate(self.clusters):
            cluster_rules = self.__get_rules(cluster)
            cluster_rules.sort(key=len)
            
            for i, rule in enumerate(cluster_rules):
                a, b = rule
                ab = a + b
                
                if not is_unique(a, b):
                    continue
                if any([set(x).issubset(ab) for x in below_threshold]):
                    excluded += 1
                    continue

                # Used to avoid re-calculation of same values
                support_a = self.__support(a)
                support_b = self.__support(b)
                support_ab = self.__support(a, b)
                confidence = 0 if support_a == 0 else support_ab/support_a
                lift = 0 if support_b == 0 else confidence/support_b

                if support_ab < min_support:
                    below_threshold.add(ab)
                    ignored += 1
                    continue

                a_str = tuple(self.names[i] for i in a)
                b_str = tuple(self.names[i] for i in b)
                
                new_rule = Rule(a_str, b_str, support_ab, confidence, lift)
                self.rules.append(new_rule)
                
                if i % 10000 == 0:
                    self.__log("Iteration #{i}: {excluded:,} excluded, {ignored:,} ignored.")
        
        print(f'Finished generating rules in {(time.time() - start):,.2f} seconds')
        # TODO: Change to pickle
        if write_to_csv:
            self.__write_rules_to_csv()

        return self.rules

if __name__ == '__main__':
    arm = MSTARM('data/rust_vectors.csv', debug_mode=True)
    arm.generate_rules()



    
