use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;
use std::fmt::Debug;

#[pyclass]
pub struct RandomNumberGenerator {
    rng: StdRng,
}

#[pymethods]
impl RandomNumberGenerator {
    #[new]
    #[pyo3(signature = (seed=None))]
    pub fn new(seed: Option<u64>) -> Self {
        let rng: StdRng;
        if let Some(seed) = seed {
            rng = StdRng::seed_from_u64(seed);
        } else {
            rng = StdRng::from_os_rng();
        }
        RandomNumberGenerator { rng }
    }

    pub fn rand(&mut self) -> f32 {
        self.rng.random::<f32>()
    }

    pub fn randn(&mut self) -> f32 {
        self.rng.sample::<f32, _>(StandardNormal)
    }

    pub fn randint(&mut self, min: usize, max: usize) -> usize {
        self.rng.random_range(min..max)
    }

    pub fn weighted_categorical(&mut self, weights: Vec<f32>) -> usize {
        let weights_sum: f32 = weights.iter().sum();
        let normalized_weights: Vec<f32> = weights.iter().map(|&w| w / weights_sum).collect();

        let rand = self.rand();

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

pub trait Samplable<T>: Send + Sync + Debug {
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

#[pyclass(str = "Uniform(min={min}, max={max})")]
#[derive(Clone, Debug)]
pub struct Uniform {
    pub min: f32,
    pub max: f32,
}

impl Uniform {
    pub fn new(min: f32, max: f32) -> Self {
        Uniform { min, max }
    }
}

impl Samplable<f32> for Uniform {
    fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> f32 {
        let mut sampled: f32;
        if let Some(rng) = rng {
            sampled = rng.rand();
        } else {
            let mut rng = RandomNumberGenerator::new(None);
            sampled = rng.rand();
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
    fn pynew(min: f32, max: f32) -> Self {
        Uniform { min, max }
    }
    #[pyo3(signature=(rng=None))]
    pub fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> f32 {
        Samplable::sample(self, rng)
    }
}

#[pyclass(str = "Categorical(n_categories={n_categories})")]
#[derive(Clone, Debug)]
pub struct Categorical {
    n_categories: usize,
}

impl Samplable<usize> for Categorical {
    fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> usize {
        let rng = if let Some(rng) = rng {
            rng
        } else {
            &mut RandomNumberGenerator::new(None)
        };
        rng.randint(0, self.n_categories)
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

#[pyclass(str = "Normal(mean={mean}, std={std})")]
#[derive(Clone, Debug)]
pub struct Normal {
    mean: f32,
    std: f32,
}

impl Normal {
    fn new(&self, mean: f32, std: f32) -> Self {
        Normal { mean, std }
    }
}

impl Samplable<f32> for Normal {
    fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> f32 {
        let sampled: f32;
        if let Some(rng) = rng {
            sampled = rng.randn();
        } else {
            let mut rng = RandomNumberGenerator::new(None);
            sampled = rng.randn();
        }
        self.mean + sampled * self.std
    }
    fn clone_box(&self) -> Box<dyn Samplable<f32>> {
        Box::new(self.clone())
    }
}

#[pymethods]
impl Normal {
    #[new]
    fn pynew(mean: f32, std: f32) -> Self {
        Normal { mean, std }
    }
    #[pyo3(signature=(rng=None))]
    pub fn sample(&self, rng: Option<&mut RandomNumberGenerator>) -> f32 {
        Samplable::sample(self, rng)
    }
}

// pub struct TruncatedNormal {
//     mean: f32,
//     std: f32,
//     min: f32,
//     max: f32,
// }
