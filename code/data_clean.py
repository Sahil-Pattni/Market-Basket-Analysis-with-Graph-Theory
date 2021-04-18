import pandas as pd 
import numpy as np 
import time
import re 

# Columns to read in
fields = ['Order Number', 'Product', 'Product Category','Client City','Sale Date Time','Discount Amount']
start = time.time()
data = pd.read_csv('data/original_data.csv',sep=';', usecols=fields) # NOTE: Remove nrows for final
print(f'Finished reading in dataset in {time.time()-start:.2f} seconds')

# ----- DATA CLEANING ----- #
# Rename columns
data.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)

# lambda to clear numeric values and remove whitespace
clean = lambda x: re.sub('[0-9]+', '', x).strip().lower()

# lowercase product and remove numbers
data['product'] = data['product'].apply(clean)
# lowecase category and remove numbers
data['product_category'] = data['product_category'].apply(clean)

# Clean up city name
data['client_city'] = data['client_city'].apply(lambda x: x.lower().strip().replace(' ','_'))

# Date formatter lambda
date_format = lambda x: x.split(' ')[0].replace('-','')
# Add unique basket ID column
data['basket_id'] =  data.order_number.apply(str) + data.sale_date_time.apply(date_format)

# ----- WRITE OUT ----- #

products = data.product_category.unique()

import csv
with open('data/output/unique_products.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(['PRODUCT']) # Adding header row since Rust ignores first row
    writer.writerows([[p] for p in products])

columns_to_export = ['product', 'product_category', 'client_city', 'discount_amount', 'basket_id']
data.to_csv('data/output/original_cleaned.csv', index=False, columns=columns_to_export)

# ----- WRITE OUT (without fuel) ----- #
import csv
with open('data/output/unique_products_no_fuel.csv', 'w+') as f:
    writer = csv.writer(f)
    writer.writerow(['PRODUCT']) # Adding header row since Rust ignores first row
    rows = [[p] for p in products if p != 'fuel']
    writer.writerows(rows)

no_fuel = data[data.product_category != 'fuel']
no_fuel.to_csv('data/output/original_cleaned_no_fuel.csv', index=False, columns=columns_to_export)


# ---- Read in Rust files (RUN ONLY IF RUST VECTORS GENERATED) and pickle output ---- #
complete_filepath = 'data/rust_vectors_product_category'
no_fuel_filepath = complete_filepath + '_no_fuel'
complete = pd.read_csv(f'{complete_filepath}.csv')
complete.to_pickle(f'{complete_filepath}.pkl')
no_fuel = pd.read_csv(f'{no_fuel_filepath}.csv')
no_fuel.to_pickle(f'{no_fuel_filepath}.pkl')