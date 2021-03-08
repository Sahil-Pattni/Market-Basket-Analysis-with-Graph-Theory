import csv
import time
import pandas as pd
from efficient_apriori import apriori, itemsets



def get_apriori_rules(min_support=0.1, min_confidence=0.2):
    with open('data/purchases.csv', 'r') as f:
        reader = csv.reader(f)
        transactions = [tuple(row) for row in reader]
    _, rules = apriori(transactions, min_support=min_support, min_confidence=min_confidence)
    # Sort by confidence
    rules.sort(key=lambda x: x.confidence, reverse=True)

    return rules