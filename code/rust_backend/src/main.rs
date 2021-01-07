use csv::Reader;
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

fn make_vectors(path: &str) -> Result<HashMap<u32, [i8; 45]>, Box<dyn Error>> {
    let products = import_unqiue("../data/unique.csv")?; // unique products
    let mut transactions: HashMap<u32, [i8; 45]> = HashMap::new();
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



fn main() {
    let _filepath = "../data/products.csv";
    let start = Instant::now();
    let mut num_baskets = 0;
    match make_vectors(_filepath) {
        Ok(res) => {
            num_baskets = res.iter().len();
            println!("There are {} baskets", num_baskets);
        },
        Err(err) => println!("Error: {:?}", err)
    };
    let duration = start.elapsed().as_secs();
    println!("Finished generating {} baskets in {} seconds", num_baskets, duration);
}
