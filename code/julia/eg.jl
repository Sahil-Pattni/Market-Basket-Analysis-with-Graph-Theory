Threads.nthreads()

# Import packages
using DataFrames, CSV, JDF, Formatting, Statistics

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
    res = Int64[]
    for i in 1:size(products, 1)
        push!(res, products[i] in transaction ? 1 : 0)
    end
    print(typeof(res))
    return res
end


seconds_elapsed = @timed begin
    # vectorized function
    vectors = gen_vector.(baskets[1:5])
end

vectors = Array(vectors)
printfmt("It took {1:.2f} minutes to get the vectors from {2:d} baskets (gen, vectorized)", seconds_elapsed[2]/60, size(baskets, 1))
vectors
typeof(vectors[1])
shape(vectors)
cor(vectors)

cmat[2]
