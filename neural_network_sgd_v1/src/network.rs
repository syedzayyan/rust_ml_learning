use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand_distr::StandardNormal;

use std::fs::File;
use std::io::Write;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

fn to_tuple(inp: &[usize]) -> (usize, usize) {
    match inp {
        [a, b] => (*a, *b),
        _ => panic!(),
    }
}

fn zero_vec_like(arr: &[Array2<f64>]) -> Vec<Array2<f64>> {
    arr.iter()
        .map(|x| Array2::zeros(to_tuple(x.shape())))
        .collect()
}

pub struct Network {
    num_layers: usize,
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub fn init_param(sizes: &[usize]) -> Network {
        let num_layers = sizes.len();
        let mut biases: Vec<Array2<f64>> = Vec::new();
        let mut weights: Vec<Array2<f64>> = Vec::new();

        for i in 1..num_layers {
            biases.push(Array::random((sizes[i], 1), StandardNormal));
            weights.push(Array::random((sizes[i], sizes[i - 1]), StandardNormal));
        }

        Network {
            num_layers: num_layers,
            biases: biases,
            weights: weights,
        }
    }

    pub fn forward_prop(&mut self, input: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut activations: Vec<Array2<f64>> = Vec::new();
        activations.push(input.clone());
        let mut zs: Vec<Array2<f64>> = Vec::new();

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let z = w.dot(activations.last().unwrap()) + b;
            zs.push(z.clone());
            activations.push(z.mapv(|x| sigmoid(x)));
        }
        return (activations, zs);
    }

    pub fn backprop(
        &mut self,
        output: &Array2<f64>,
        activations: &Vec<Array2<f64>>,
        zs: &Vec<Array2<f64>>,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut d_b: Vec<Array2<f64>> = zero_vec_like(&self.biases);
        let mut d_w: Vec<Array2<f64>> = zero_vec_like(&self.weights);

        // backwards pass
        let mut delta = self.cost_func(activations.last().unwrap(), output)
            * zs.last().unwrap().mapv(|x| sigmoid_derivative(x));

        *d_b.last_mut().unwrap() = delta.clone();
        *d_w.last_mut().unwrap() =
            delta.dot(&activations[activations.len() - 2].clone().reversed_axes());

        for l in 2..self.num_layers {
            let z = &zs[zs.len() - l];
            let sp = z.mapv(|x| sigmoid_derivative(x));
            delta = self.weights[self.weights.len() - l + 1]
                .clone()
                .reversed_axes()
                .dot(&delta)
                * sp;

            let len_nb = d_b.len();
            let len_nw = d_w.len();
            d_b[len_nb - l] = delta.clone();
            d_w[len_nw - l] = delta.dot(
                &activations[activations.len() - l - 1]
                    .clone()
                    .reversed_axes(),
            );
        }

        (d_b, d_w)
    }

    pub fn update_params(
        &mut self,
        d_w: &Vec<Array2<f64>>,
        d_b: &Vec<Array2<f64>>,
        learning_rate: f64,
    ) {
        let weights = self.weights.clone();
        let bias = self.biases.clone();

        self.weights = weights
            .iter()
            .zip(d_w.iter())
            .map(|(x, y)| x - (y * learning_rate))
            .collect();
        self.biases = bias
            .iter()
            .zip(d_b.iter())
            .map(|(x, y)| x - (y * learning_rate))
            .collect();
    }

    pub fn save_weights(&self, file_path: &str) -> std::io::Result<()> {
        let weights_str = format!("{:?}", self.weights);
        let biases_str = format!("{:?}", self.biases);

        let mut file = File::create(file_path)?;

        file.write_all(b"Weight\n")?;
        file.write_all(weights_str.as_bytes())?;
        file.write_all(b"\nBias\n")?;
        file.write_all(biases_str.as_bytes())?;

        Ok(())
    }

    fn cost_func(&mut self, output_activations: &Array2<f64>, output: &Array2<f64>) -> Array2<f64> {
        output_activations - output
    }


    
}
