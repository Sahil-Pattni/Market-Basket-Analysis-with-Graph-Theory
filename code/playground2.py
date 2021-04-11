import arm_cython as cy
import time

import arm
from timeit import timeit
if __name__ == '__main__':
    csv_filepath = 'data/rust_vectors_product_category.csv'
    pickle_filepath = 'data/rust_vectors_product_category.pkl'
    

    # start = time.time()
    # cy.init(csv_filepath, pickled=False, graph_exists=False)
    # cy.generate_rules()
    # cy_time = time.time() - start

    start = time.time()
    cy.init(pickle_filepath, pickled=True, graph_exists=True)
    rules = cy.generate_rules(min_support=0.005, min_confidence=0.3)
    cy_optimized_time = time.time() - start

    # start = time.time()
    # py = arm.MSTARM(csv_filepath)
    # py.generate_rules()
    # py_time = time.time() - start


    # print(f'Python: {py_time:,.4f} seconds')
    # print(f'Cython: {cy_time:,.4f} seconds')
    print(f'Cython (Optimized): {cy_optimized_time:,.4f} seconds. {len(rules)} rules.')
    for rule in rules:
        print(rule)