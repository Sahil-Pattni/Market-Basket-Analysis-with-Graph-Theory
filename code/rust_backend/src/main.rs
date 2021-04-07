use csv::Reader;
use csv::Writer;
use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

// Import list of unique products to maintain uniform ordering for vectors
fn import_unqiue(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut reader = Reader::from_path(path)?;
    let mut v: Vec<String> = Vec::new(); // Vector of unique products
    for result in reader.records() {
        let _record = result?;
        v.push(_record[0].to_string());
    }
    Ok(v) // return with Result wrapper
}

fn make_vectors(path: &str) -> Result<HashMap<u64, [u8; 45]>, Box<dyn Error>> {
    // Get list of unique products
    let products = import_unqiue("../data/output/unique_products.csv")?; // unique products
    // Hashmap with each key being a basket ID
    let mut transactions: HashMap<u64, [u8; 45]> = HashMap::new();
    // CSV Reader
    let mut reader = Reader::from_path(path)?;

    for result in reader.records() {
        let record = result?; // error check
        let product = record[0].to_string();
        let basket_id: u64 = record[4].parse()?;

        // Insert basket_id if doesn't exist
        if !transactions.contains_key(&basket_id) {
            // Insert blank binary purchase vector
            transactions.insert(basket_id, [0; 45]);
        }
    
        // Vector for given basket_id
        let v = transactions.get_mut(&basket_id).unwrap();

        // Boolean mask on vector
        for (i, item) in products.iter().enumerate() {
            if item == &product {
                v[i] = 1;
            }
        }
    }
    Ok(transactions) // return type with Ok signature
}


fn write_data(path: &str, map: HashMap<u64, [u8; 45]>) -> Result<(), Box<dyn Error>> {
    let mut writer = Writer::from_path(path)?;
    // Products for header
    let products = import_unqiue("../data/output/unique_products.csv")?; // unique products
    writer.write_record(products)?;

    for (_, val) in map.iter() {
        let mut temp_vec = Vec::new();
        for v in val {
            temp_vec.push(v.to_string());
        }
        match writer.write_record(temp_vec) {
            Ok(emp) => emp,
            Err(error) => panic!("{}", error)
        };
    };
    Ok(())
}


fn main() {
    let _filepath = "../data/output/original_cleaned.csv";
    let start = Instant::now();
    let mut num_baskets = 0;
    match make_vectors(_filepath) {
        Ok(res) => {
            num_baskets = res.iter().len();
            println!("There are {} baskets", num_baskets);
            write_data("../data/rust_vectors_temp.csv", res);
        },
        Err(err) => println!("Error: {:?}", err)
    };
    let duration = start.elapsed().as_secs();
    println!("Finished generating {} baskets in {} seconds", num_baskets, duration);
}
