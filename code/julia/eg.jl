Threads.nthreads()

# Import packages
using DataFrames, CSV, JDF, Formatting

# Change path
cd("/Users/sloth_mini/Documents/4.1/dissertation/code/")
pwd()
# Read CSV
read_time = @timed begin
     df = CSV.read("data/products.csv", DataFrame)
end

printfmt("Read CSV in {1:.2f} seconds.", read_time[2])
printfmt("CSV shape: {1:s}", size(df))

# Limit DataFrame to 1,000,000 entries.
df = df[1:1000000, :]
printfmt("CSV shape: {1:s}", size(df))

# Get unique products and baskets
products = unique(df.product)
baskets = unique(df.basket_id)
printfmt("There are {2:d} unique baskets and {1:d} products.", size(products, 1), size(baskets, 1))

# Linear function to generate vector for basket id
function gen_vector(basket_id)
    transaction = df[df.basket_id .== basket_id, 1]
    return [product in transaction ? 1 : 0 for product in products]
end

# Linear Approach
seconds_elapsed = @timed begin
    vectors = [gen_vector(bid) for bid in baskets]
end

printfmt("It took {1:.2f} minutes to get the vectors from {2:d} baskets", seconds_elapsed[2]/60, size(baskets, 1))
