// https://github.com/ryanbarouki/rust-neural-network/blob/main/src/network.rs
//https://ngoldbaum.github.io/posts/loading-mnist-data-in-rust/

mod load_data;
mod network;

use load_data::load_data_set;
use ndarray::{Array2};
use network::Network;

fn main() {
    let mnist_data = load_data_set("t10k");

    match mnist_data {
        Ok(mnist) => {
            let images: Vec<Array2<f64>> = mnist.iter().map(|data| data.image.clone()).collect();
            let classifications: Vec<Array2<f64>> = mnist
                .iter()
                .map(|data| {
                    let mut class_vec = Array2::zeros((10, 1));
                    class_vec[[data.classification as usize, 0]] = 1.0;
                    class_vec
                })
                .collect();
            let mut nn = Network::init_param(&[images[0].len(), 100usize, 50usize, classifications[0].len()]);
                for i in 1..2 {
                    for (image, classification) in images.iter().zip(classifications.iter()) {
                        let (activations, zs) = nn.forward_prop(image);
                        let (d_b, d_w) = nn.backprop(classification, &activations, &zs);
                        nn.update_params(&d_w, &d_b, 0.0001);
                    }
                    println!("Iteration {:?}", i);
                }
                let (_, zs) = nn.forward_prop(&images[0]);
                println!("{:?}", zs[zs.len() - 1]);

                nn.save_weights("weights.txt").expect("Failed to save weights");
        }
        Err(err) => {
            panic!("Error loading data set: {:?}", err)
        }
    };
}
