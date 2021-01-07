use csv::Reader;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::iter::FromIterator;
use std::time::{Duration, Instant};

fn gen_product_set(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut reader = Reader::from_path(path)?;
    let mut product_set: HashSet<String> = HashSet::new();
    

    for result in reader.records() {
        let _record = match result {
            Ok(result) => {
                product_set.insert(result[0].to_string());
                continue;
            },
            Err(error) => println!("Encountered error reading CSV. {}", error)
        };

    }
    let v: Vec<String> = Vec::from_iter(product_set);
    Ok(v) // return 
}

fn gen_vectors(path: &str) -> Result<HashMap<[i8; 45], i32>, Box<dyn Error>> {
    let mut transactions = HashMap::new();

    let mut reader = Reader::from_path(path)?;

    // Read and print headers
    let headers = reader.headers()?;
    println!("{:?}", headers);

    // Fill in hashmap
    for (i, result) in reader.records().enumerate() {
        if i > 5 {break;} // temporary, take this out
        let result = &result?; // error check cast
        println!("{:?}", &result[0]);
        // TODO: Add to hashmap and perform binary fill
    }
    println!("\n");
    for (i, result) in reader.records().enumerate() {
        if i > 5 {break;} // temporary, take this out
        let result = &result?; // error check cast
        println!("{:?}", &result[0]);
        // TODO: Add to hashmap and perform binary fill
    }

    Ok(transactions) // return type with Ok signature
}

fn main() {
    let _filepath = "../data/products.csv";
    let start = Instant::now();
    let result = gen_product_set(_filepath);
    let duration = start.elapsed().as_secs();

    println!("Finished in {} seconds", duration);
    println!("{:?}", result);
}
