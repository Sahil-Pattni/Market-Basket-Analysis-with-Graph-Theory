import csv
import time
import pandas as pd
from efficient_apriori import apriori, itemsets

transactions = []
with open('data/purchases.csv', 'r') as f:
    reader = csv.reader(f)
    #transactions = [tuple(row) for row in reader]
    i = 0
    for row in reader:
        transactions.append(tuple(row))
        i += 1
        if i > 2000:
            break


itemset, rules = apriori(transactions, min_support=0.1, min_confidence=0.2)

rules.sort(key=lambda x: x.confidence, reverse=True)

for rule in rules:
    print(f'[{rule.confidence}] {rule}')
