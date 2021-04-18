# %%
import matplotlib.pyplot as plt
from math import factorial
import pandas as pd

# --- Plot out number of potential association rules --- #
def num_combinations(n, k):
    return factorial(n)/(factorial(k) * factorial(n-k))


def num_rule(d):
    num = 0 # number of rules
    for k in range(1, d):
        num += num_combinations(d, k) * sum([num_combinations(d-k, j) for j in range(1, (d-k)+1)])
    return num


values = [num_rule(n) for n in range(2, 15)]
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(values, '--o', color='k')
ax.set_xlabel("Length of Itemset (d)")
ax.set_ylabel("Number of Rules")
ax.set_yticklabels([f'{int(y):,}' for y in ax.get_yticks().tolist()])
plt.savefig('../images/numrules.png')


# ----- Plot product category distribution ----- #
# %%
df = pd.read_pickle('data/rust_vectors_product_category.pkl')

actual = df.shape[0]
results = []
for col in df.columns:
    filt = df[df[col] == 1]
    filt_size = filt.shape[0]
    results.append([col, filt_size/actual * 100])
results.sort(key=lambda x: x[1], reverse=True)

# %%
fig, ax = plt.subplots(figsize=(10,6))
names = [r[0] for r in results]
vals = [r[1] for r in results]
ax.bar(names, vals, color='0.25')
ax.set_xticklabels(names, rotation=90)
ax.set_yticklabels([f'{t}%' for t in ax.get_yticks()])
ax.set_ylabel('% of transactions present')
plt.tight_layout()
plt.savefig('../images/category_dist.png')
# %%
