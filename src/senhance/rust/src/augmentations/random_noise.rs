use crate::audio::Audio;
use crate::augmentations::augmentation::RandomAugmentation;
use crate::augmentations::distributions::{RandomNumberGenerator, Samplable, Uniform};
use ndarray::{Array, Array2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

#[pyclass]
#[derive(Clone)]
pub struct RandomNoiseParameters {
    noise: Audio,
    #[pyo3(get)]
    amplitude: f32,
    #[pyo3(get)]
    beta: f32,
}

#[pyclass]
#[derive(Debug)]
pub struct RandomNoise {
    amplitude_distribution: Box<dyn Samplable<f32>>,
    beta_distribution: Box<dyn Samplable<f32>>,
    #[pyo3(get)]
    p: f32,
}

fn colored_noise(
    amplitude: f32,
    beta: f32,
    size: usize,
    rng: Option<&mut RandomNumberGenerator>,
) -> Array2<f32> {
    let rng = if let Some(rng) = rng {
        rng
    } else {
        &mut RandomNumberGenerator::new(None)
    };

    let noise: Vec<f64> = (0..size).map(|_| rng.randn() as f64).collect();
    let mut complex_data: Vec<Complex<f64>> = noise.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // fft forward
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(size);
    fft.process(&mut complex_data);

    // Apply 1/f^beta filter
    for i in 0..size {
        // Calculate frequency (normalized)
        let f = if i <= size / 2 {
            i as f64 / size as f64
        } else {
            (size - i) as f64 / size as f64
        };

        // Apply filter
        let filter = f.max(1e-8).powf(-beta as f64 / 2.0);
        complex_data[i] = complex_data[i] * filter;
    }

    // fft inverse
    let ifft = planner.plan_fft_inverse(size);
    ifft.process(&mut complex_data);

    let mut noise: Vec<f64> = complex_data
        .iter()
        .map(|&c| c.re / (size as f64).sqrt()) // Normalize by sqrt(size)
        .collect();

    // Normalize to unit standard deviation
    let mean = noise.iter().sum::<f64>() / noise.len() as f64;
    let std = (noise.iter().map(|&x| x * x).sum::<f64>() / noise.len() as f64).sqrt();
    for val in &mut noise {
        *val = amplitude as f64 * (*val - mean) / std;
    }

    let result = Array::from_vec(noise)
        .into_shape_with_order((1, size))
        .expect("");
    result.mapv(|x| x as f32)
}

impl RandomNoise {
    pub fn new(
        amplitude_distribution: Box<dyn Samplable<f32>>,
        beta_distribution: Box<dyn Samplable<f32>>,
        p: f32,
    ) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be between 0 and 1");
        RandomNoise {
            amplitude_distribution,
            beta_distribution,
            p,
        }
    }
}

impl Clone for RandomNoise {
    fn clone(&self) -> Self {
        RandomNoise {
            amplitude_distribution: self.amplitude_distribution.clone_box(),
            beta_distribution: self.beta_distribution.clone_box(),
            p: self.p,
        }
    }
}

impl RandomAugmentation for RandomNoise {
    type Parameters = RandomNoiseParameters;
    fn name(&self) -> &str {
        "random_noise"
    }
    fn p(&self) -> f32 {
        self.p
    }
    fn sample_parameters(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> RandomNoiseParameters {
        let amplitude = self.amplitude_distribution.sample(rng.as_deref_mut());
        let beta = self.beta_distribution.sample(rng.as_deref_mut());
        let noise = colored_noise(amplitude, beta, audio.waveform.shape()[1], rng);
        let aud = Audio::new(noise, audio.sample_rate);
        println!("{:?}", amplitude);
        RandomNoiseParameters {
            noise: aud,
            amplitude,
            beta,
        }
    }
    fn augment(&self, waveform: &Array2<f32>, parameters: &RandomNoiseParameters) -> Array2<f32> {
        waveform + &parameters.noise.waveform
    }
}

#[pymethods]
impl RandomNoise {
    #[new]
    #[pyo3(signature = (min_amplitude, max_amplitude, min_beta, max_beta, p=1.0))]
    fn pynew(
        min_amplitude: f32,
        max_amplitude: f32,
        min_beta: f32,
        max_beta: f32,
        p: f32,
    ) -> PyResult<Self> {
        let amplitude_distribution = Box::new(Uniform {
            min: min_amplitude,
            max: max_amplitude,
        });
        let beta_distribution = Box::new(Uniform {
            min: min_beta,
            max: max_beta,
        });
        Ok(RandomNoise {
            amplitude_distribution,
            beta_distribution,
            p,
        })
    }
    #[pyo3(signature = (audio, rng=None))]
    fn sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> RandomNoiseParameters {
        RandomAugmentation::sample_parameters(self, audio, rng)
    }
    fn augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &RandomNoiseParameters,
    ) -> Py<PyArray2<f32>> {
        let augmented =
            RandomAugmentation::augment(self, &waveform.as_array().to_owned(), parameters);
        PyArray2::from_array(py, &augmented).to_owned().into()
    }
}
