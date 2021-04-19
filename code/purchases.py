"""
Writes out the string representation of the binary purchase vectors for the Apriori algorithm
to use.
"""

import os
import csv
import time
import pandas as pd

df = pd.read_pickle('data/rust_vectors_product_category_no_fuel.pkl')
columns = df.columns
purchases = []
start = time.time()

limit = 100000
for i in range(df.shape[0]):
    current_purchase = []
    for j in range(df.shape[1]):
        if df.iloc[i,j] == 1:
            current_purchase.append(columns[j])
    purchases.append(current_purchase)

end = time.time() - start
os.system('say "your code has finished running"')
print(f'Finished getting purchases in {end/60:,.2f} minutes.')
estimated_time = (end/limit) * df.shape[0] 
print(f'Estimated completion time for full dataset is {estimated_time/60:.2f} minutes')

with open('data/purchases_no_fuel.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerows(purchases)