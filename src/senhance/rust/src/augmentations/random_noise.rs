use super::o3utils::extract_distribution;
use crate::audio::Audio;
use crate::augmentations::augmentation::{Augments, RandomAugmentation};
use crate::augmentations::distributions::{RandomNumberGenerator, Samplable};
use ndarray::{Array, Array2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

#[pyclass(str = "RandomNoiseParameters(noise={noise}, snr_db={snr_db}, beta={beta})")]
#[derive(Clone)]
pub struct RandomNoiseParameters {
    #[pyo3(get)]
    noise: Audio,
    #[pyo3(get)]
    snr_db: f32,
    #[pyo3(get)]
    beta: f32,
}

#[pyclass]
#[derive(Debug)]
pub struct RandomNoise {
    snr_db_distribution: Box<dyn Samplable<f32>>,
    beta_distribution: Box<dyn Samplable<f32>>,
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
        snr_db_distribution: Box<dyn Samplable<f32>>,
        beta_distribution: Box<dyn Samplable<f32>>,
    ) -> Self {
        RandomNoise {
            snr_db_distribution,
            beta_distribution,
        }
    }
}

impl Clone for RandomNoise {
    fn clone(&self) -> Self {
        RandomNoise {
            snr_db_distribution: self.snr_db_distribution.clone_box(),
            beta_distribution: self.beta_distribution.clone_box(),
        }
    }
}

impl Augments for RandomNoise {
    type Parameters = RandomNoiseParameters;
    fn name(&self) -> &str {
        "random_noise"
    }
    fn sample_parameters(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<RandomNoiseParameters, String> {
        let snr_db = self.snr_db_distribution.sample(rng.as_deref_mut());
        let beta = self.beta_distribution.sample(rng.as_deref_mut());
        let noise_waveform = colored_noise(1.0, beta, audio.waveform.shape()[1], rng);
        let mut noise = Audio::new(noise_waveform, audio.sample_rate);
        noise.normalize(-snr_db + audio.loudness_db().sum() / audio.loudness_db().len() as f32);
        println!("{:?}", snr_db);
        Ok(RandomNoiseParameters {
            noise,
            snr_db,
            beta,
        })
    }
    fn augment(&self, waveform: &Array2<f32>, parameters: &RandomNoiseParameters) -> Array2<f32> {
        waveform + &parameters.noise.waveform
    }
}

#[pyclass(name = "RandomNoise")]
#[derive(Debug, Clone)]
pub struct PyRandomNoise {
    pub random_noise: RandomAugmentation<RandomNoise>,
}

#[pymethods]
impl PyRandomNoise {
    #[new]
    #[pyo3(signature = (snr_db_distribution, beta_distribution, p=1.0))]
    fn pynew(
        py: Python,
        snr_db_distribution: PyObject,
        beta_distribution: PyObject,
        p: Option<f32>,
    ) -> PyResult<Self> {
        let snr_db = extract_distribution(py, snr_db_distribution)?;
        let beta = extract_distribution(py, beta_distribution)?;
        let random_noise = RandomNoise::new(snr_db, beta);
        let random_random_noise: RandomAugmentation<RandomNoise>;
        if let Some(p) = p {
            random_random_noise = RandomAugmentation::new(random_noise, p).unwrap();
        } else {
            random_random_noise = RandomAugmentation::new(random_noise, 1.0).unwrap();
        }
        Ok(PyRandomNoise {
            random_noise: random_random_noise,
        })
    }

    #[pyo3(name = "sample_parameters", signature = (audio, rng=None))]
    fn sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> PyResult<Option<RandomNoiseParameters>> {
        self.random_noise
            .sample_parameters(audio, rng)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    #[pyo3(name = "augment")]
    fn py_augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &RandomNoiseParameters,
    ) -> Py<PyArray2<f32>> {
        let augmented = self
            .random_noise
            .augment(&waveform.as_array().to_owned(), &Some(parameters.clone()));
        PyArray2::from_array(py, &augmented).to_owned().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::augmentations::distributions::Uniform;
    use rand::Rng;

    fn create_random_audio(n_samples: usize, sr: usize) -> Audio {
        let mut rng = rand::rng();
        let rand_waveform = Array2::from_shape_fn((1_usize, n_samples as usize), |_| rng.random());
        Audio {
            waveform: rand_waveform,
            sample_rate: Some(sr),
        }
    }

    #[test]
    fn test_random_noise() {
        const SR: usize = 16_000;
        let random_audio = create_random_audio(3 * SR, SR);
        let snr_db_distribution = Box::new(Uniform::new(5.0, 25.0));
        let beta_distribution = Box::new(Uniform::new(-2.0, 2.0));
        let random_noise = RandomNoise::new(snr_db_distribution, beta_distribution);

        let mut rng = RandomNumberGenerator::new(Some(0));
        let parameters = random_noise
            .sample_parameters(&random_audio, Some(&mut rng))
            .unwrap();
        let noisy = random_noise.augment(&random_audio.waveform, &parameters);
        assert_ne!(
            noisy, random_audio.waveform,
            "Noisy audio should be different from original"
        );
    }

    #[test]
    fn test_norandom_noise() {
        const SR: usize = 16_000;
        let random_audio = create_random_audio(3 * SR, SR);
        let snr_db_distribution = Box::new(Uniform::new(5.0, 25.0));
        let beta_distribution = Box::new(Uniform::new(-2.0, 2.0));
        let random_noise = RandomAugmentation::new(
            RandomNoise::new(snr_db_distribution, beta_distribution),
            0.0,
        )
        .unwrap();

        let mut rng = RandomNumberGenerator::new(Some(0));
        let parameters = random_noise
            .sample_parameters(&random_audio, Some(&mut rng))
            .unwrap();
        let noisy = random_noise.augment(&random_audio.waveform, &parameters);
        assert_eq!(
            noisy, random_audio.waveform,
            "Noisy audio should be different from original"
        );
    }
}
