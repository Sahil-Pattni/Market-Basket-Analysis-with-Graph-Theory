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

fn make_vectors(path: &str) -> Result<HashMap<u32, [u8; 45]>, Box<dyn Error>> {
    // Get list of unique products
    let products = import_unqiue("../data/unique.csv")?; // unique products
    // Hashmap with each key being a basket ID
    let mut transactions: HashMap<u32, [u8; 45]> = HashMap::new();
    // CSV Reader
    let mut reader = Reader::from_path(path)?;

    for result in reader.records() {
        let record = result?; // error check
        let product = record[0].to_string();
        let basket_id: u32 = record[1].parse()?;

        // Insert basket_id if doesn't exist
        if !transactions.contains_key(&basket_id) {
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

fn write_purchases(in_path: &str, out_path: &str) -> Result<(), Box<dyn Error>>{
    
    // Contains Vec<String> of purchases
    let mut purchases: Vec<Vec<String>> = Vec::new();
    // CSV Reader
    let mut reader = Reader::from_path(in_path)?;
    // CSV Writer
    let mut writer = Writer::from_path(out_path)?;
    // Import column names
    let products = import_unqiue("../data/columns.csv")?; // unique products
    // Iterate and add products
    let data = reader.records();
    for row in data {
        let mut current_purchase: Vec<String> = Vec::new();
        let row = row.iter();
        for (i, item) in row.enumerate() {
            let item: u8 = item[i].parse()?;
            if (item as u8) == (1 as u8) {
                current_purchase.push(products[i].clone());
            }
        }
        purchases.push(current_purchase);
    }

    // Write to csv
    for row in purchases {
        writer.write_record(row)?;
    }

    Ok(())
}

fn write_data(path: &str, map: HashMap<u32, [u8; 45]>) -> Result<(), Box<dyn Error>> {
    let mut writer = Writer::from_path(path)?;
    // Products for header
    let products = import_unqiue("../data/unique.csv")?; // unique products
    writer.write_record(products)?;
    for (_, val) in map.iter() {
        let mut temp_vec = Vec::new();
        for v in val {
            temp_vec.push(v.to_string());
        }
        writer.write_record(temp_vec)?;
    };
    Ok(())
}


fn main() {
    //let _filepath = "../data/products.csv";
    //let start = Instant::now();
    //let mut num_baskets = 0;
    // match make_vectors(_filepath) {
    //     Ok(res) => {
    //         num_baskets = res.iter().len();
    //         println!("There are {} baskets", num_baskets);
    //         write_data("../data/rust_vectors_temp.csv", res);
    //         //write_purchases("../data/purchases.csv", res);
    //     },
    //     Err(err) => println!("Error: {:?}", err)
    // };
    
    let _in_path = "../data/rust_vectors.csv";
    let _out_path = "../data/output/purchases.csv";
    match write_purchases(_in_path, _out_path) {
        Ok(()) => {()}
        Err(err) => println!("Error: {:?}", err)
    }
    //let duration = start.elapsed().as_secs();
    //println!("Finished generating {} baskets in {} seconds", num_baskets, duration);
}
