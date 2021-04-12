import arm_cython as cy
import time

import arm
from timeit import timeit
if __name__ == '__main__':
    csv_filepath = 'data/rust_vectors_product_category_no_fuel.csv'
    pickle_filepath = 'data/rust_vectors_product_category_no_fuel.pkl'
    old_filepath = 'data/rust_vectors.csv'

    min_support = 0.0001
    min_confidence = 0.25
    
    # start = time.time()
    # py = arm.MSTARM(csv_filepath)
    # rules = py.generate_bicluster_rules(min_support=min_support, min_confidence=min_confidence)
    # py_time = time.time() - start
    # print(f'Python: {py_time:,.4f} seconds. {len(rules):,} rules.')

    
    start = time.time()
    cy.init(csv_filepath, pickled=False, graph_exists=False)
    rules = cy.generate_bicluster_rules(min_support=min_support, min_confidence=min_confidence)
    cy_time = time.time() - start
    print(f'Cython: {cy_time:,.4f} seconds. {len(rules):,} rules. ')

    # start = time.time()
    # cy.init(pickle_filepath, pickled=True, graph_exists=True)
    # rules = cy.generate_rules(min_support=min_support, min_confidence=min_confidence)
    # cy_optimized_time = time.time() - start
    # print(f'Cython (Optimized): {cy_optimized_time:,.4f} seconds. {len(rules):,} rules.')

    if len(rules) > 10:
        for rule in rules[:10]:
            print(rule)
    else:
        for rule in rules:
            print(rule)
    