use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

#[pyclass]
pub struct RandomNumberGenerator {
    rng: StdRng,
}

//#[pymethods]
impl RandomNumberGenerator {
    //#[new]
    pub fn new(seed: Option<u64>) -> Self {
        let rng: StdRng;
        if let Some(seed) = seed {
            rng = StdRng::seed_from_u64(seed);
        } else {
            rng = StdRng::from_os_rng();
        }
        RandomNumberGenerator { rng }
    }
    pub fn random(&mut self) -> f32 {
        self.rng.random::<f32>()
    }

    pub fn randn(&mut self) -> f32 {
        self.rng.sample::<f32, _>(StandardNormal)
    }

    pub fn categorical(&mut self, n: usize) -> usize {
        self.rng.random_range(0..n)
    }

    pub fn weighted_categorical(&mut self, weights: Vec<f32>) -> usize {
        let weights_sum: f32 = weights.iter().sum();
        let normalized_weights: Vec<f32> = weights.iter().map(|&w| w / weights_sum).collect();

        let rand = self.random();

        let mut cumsum: Vec<f32> = Vec::with_capacity(normalized_weights.len());
        let mut running_sum = 0.0;

        for weight in normalized_weights.iter() {
            running_sum += weight;
            cumsum.push(running_sum);
        }

        for (i, &cum_prob) in cumsum.iter().enumerate() {
            if rand <= cum_prob {
                return i;
            }
        }
        normalized_weights.len() - 1
    }
}

pub trait Samplable<T>: Send + Sync {
    fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> T;
    fn clone_box(&self) -> Box<dyn Samplable<T>>;
}

//impl<T, S> Samplable<T> for S
//where
//    S: 'static + Samplable<T> + Clone,
//    T: 'static,
//{
//    fn clone_box(&self) -> Box<dyn Samplable<T>> {
//        Box::new(self.clone())
//    }
//}

#[pyclass(str = "uniform(min={min}, max={max})")]
#[derive(Clone, Debug)]
pub struct Uniform {
    pub min: f32,
    pub max: f32,
}

impl Samplable<f32> for Uniform {
    fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> f32 {
        let mut sampled: f32;
        if let Some(rng) = rng {
            sampled = rng.random();
        } else {
            let mut rng = rand::rng();
            sampled = rng.random::<f32>();
        }
        sampled = self.min + sampled * (self.max - self.min);
        sampled
    }
    fn clone_box(&self) -> Box<dyn Samplable<f32>> {
        Box::new(self.clone())
    }
}

#[pymethods]
impl Uniform {
    #[new]
    fn new(min: f32, max: f32) -> Self {
        Uniform { min, max }
    }
    #[pyo3(signature=(rng=None))]
    pub fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> f32 {
        Samplable::sample(self, rng)
    }
}

#[pyclass(str = "categorical(n_categories={n_categories})")]
#[derive(Clone, Debug)]
pub struct Categorical {
    n_categories: usize,
}

impl Samplable<usize> for Categorical {
    fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> usize {
        if let Some(rng) = rng {
            rng.categorical(self.n_categories)
        } else {
            let mut rng = rand::rng();
            rng.random_range(0..self.n_categories)
        }
    }
    fn clone_box(&self) -> Box<dyn Samplable<usize>> {
        Box::new(self.clone())
    }
}

#[pymethods]
impl Categorical {
    #[new]
    fn pynew(n_categories: usize) -> Self {
        Categorical { n_categories }
    }
    #[pyo3(signature=(rng=None))]
    pub fn pysample(&self, rng: Option<&mut RandomNumberGenerator>) -> usize {
        Samplable::sample(self, rng)
    }
}

#[pyclass(str = "WeightedCategorical(weights=)")]
#[derive(Clone, Debug)]
pub struct WeightedCategorical {
    weights: Vec<f32>,
}

impl WeightedCategorical {
    pub fn new(weights: Vec<f32>) -> Self {
        WeightedCategorical { weights }
    }
}

impl Samplable<usize> for WeightedCategorical {
    fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> usize {
        let rng = if let Some(rng) = rng {
            rng
        } else {
            &mut RandomNumberGenerator::new(None)
        };
        rng.weighted_categorical(self.weights.clone())
    }
    fn clone_box(&self) -> Box<dyn Samplable<usize>> {
        Box::new(self.clone())
    }
}

#[pymethods]
impl WeightedCategorical {
    #[new]
    fn pynew(weights: Vec<f32>) -> Self {
        WeightedCategorical { weights }
    }
    #[pyo3(signature=(rng=None))]
    pub fn pysample(&self, rng: Option<&mut RandomNumberGenerator>) -> usize {
        Samplable::sample(self, rng)
    }
}

// pub struct Gaussian {
//     distribution: Box<Normal>,
// }

// impl Gaussian {
//     fn new(&self, mean: f32, std: f32) -> Self {
//         let distribution = Normal::new(mean, std).unwrap();
//         Gaussian { distribution }
//     }
// }

// impl Samplable for Gaussian {
//     fn sample(&self, rng: &mut StdRng) -> f32 {
//         self.distribution.sample(rng)
//     }
// }

// pub struct TruncatedNormal {
//     mean: f32,
//     std: f32,
//     min: f32,
//     max: f32,
// }
