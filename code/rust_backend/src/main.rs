use csv::Reader;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::iter::FromIterator;
use std::time::Instant;

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

fn import_unqiue(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    println!("Started unique import....");
    let mut reader = Reader::from_path(path)?;
    let mut v: Vec<String> = Vec::new();
    println!("Started reading unique....");
    // for result in reader.records() {
    //     match result {
    //         Ok(record) => {
    //             v.push(record[0].to_string());
    //         },
    //         Err(error) => println!("Error!")
    //     }
    // }
    for result in reader.records() {
        let _record = result?;
        v.push(_record[0].to_string());
    }
    println!("Writing out unique....");
    Ok(v) // return type
}

fn untitled(path: &str) -> Result<HashMap<u32, [i8; 45]>, Box<dyn Error>> {
    println!("Started main code...");
    let products = import_unqiue("../data/unique.csv")?; // unique products
    println!("Finished getting unique products");
    println!("Initializing vars...");
    let mut transactions: HashMap<u32, [i8; 45]> = HashMap::new();
    let mut reader = Reader::from_path(path)?;

    // // Read and print headers
    // let headers = reader.headers()?;
    // println!("{:?}", headers);
    println!("Traversing...");
    let mut i = 0;
    for result in reader.records() {
        if i > 100 {break;} // temp, remove this
        i += 1;
        let record = result?; // error check
        let product = record[0].to_string();
        let basket_id: u32 = record[1].parse()?;

        // Insert basket_id if doesn't exist
        if !transactions.contains_key(&basket_id) {
            transactions.insert(basket_id, [0; 45]);
        }
        
        // Vector for given basket_id
        let v = transactions.get_mut(&basket_id).unwrap();

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
    match untitled(_filepath) {
        Ok(res) => {
            for r in res {
                println!("{:?}", r);
            }
        },
        Err(err) => println!("Error: {:?}", err)
    };
    // let start = Instant::now();
    // let result = gen_product_set(_filepath);
    // let duration = start.elapsed().as_secs();

    // for (i, item) in result.iter().enumerate() {
    //     println!("{:?}", item);
    // }


    // println!("Finished in {} seconds", duration);
    // println!("{:?}", result);
}
