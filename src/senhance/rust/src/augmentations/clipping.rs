use crate::audio::Audio;
use crate::augmentations::augmentation::RandomAugmentation;
use crate::augmentations::distributions::{RandomNumberGenerator, Samplable};
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

use super::o3utils::extract_distribution;

#[pyclass]
#[derive(Clone)]
pub struct ClippingParameters {
    #[pyo3(get)]
    clip_percentile: f32,
}

#[pyclass]
#[derive(Debug)]
pub struct Clipping {
    quantile_distribution: Box<dyn Samplable<f32>>,
    #[pyo3(get)]
    p: f32,
}

fn clip(waveform: &Array2<f32>, q: f32) -> Result<Array2<f32>, String> {
    if q < 0.0 || q > 1.0 {
        return Err("q must be between 0 and 1.".to_string());
    }

    let mut clipped = Array2::zeros(waveform.dim());

    let mut abs = Vec::with_capacity(clipped.ncols());
    for i in 0..waveform.nrows() {
        abs.clear();
        for &sample in waveform.row(i).iter() {
            abs.push(sample.abs())
        }
        let quantile_idx = (q * abs.len() as f32).floor() as usize;
        abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let quantile = abs[quantile_idx];

        for j in 0..waveform.ncols() {
            let sample = waveform[[i, j]];
            clipped[[i, j]] = sample.max(-quantile).min(quantile);
        }
    }
    Ok(clipped)
}

impl Clipping {
    pub fn new(quantile_distribution: Box<dyn Samplable<f32>>, p: f32) -> Result<Self, String> {
        if p < 0.0 || p > 1.0 {
            return Err("p must be between 0 and 1.".to_string());
        }
        Ok(Clipping {
            quantile_distribution,
            p,
        })
    }
    fn clip(&self, waveform: &Array2<f32>, q: f32) -> Array2<f32> {
        clip(waveform, q).expect("Error when clipping")
    }
}

impl Clone for Clipping {
    fn clone(&self) -> Self {
        Clipping {
            quantile_distribution: self.quantile_distribution.clone_box(),
            p: self.p,
        }
    }
}

impl RandomAugmentation for Clipping {
    type Parameters = ClippingParameters;
    fn name(&self) -> &str {
        "clipping"
    }
    fn p(&self) -> f32 {
        self.p
    }
    fn sample_parameters(
        &self,
        _audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> ClippingParameters {
        let clip_percentile = self.quantile_distribution.sample(rng);
        println!("{:?}", clip_percentile);
        ClippingParameters { clip_percentile }
    }
    fn augment(&self, waveform: &Array2<f32>, parameters: &ClippingParameters) -> Array2<f32> {
        self.clip(waveform, parameters.clip_percentile)
    }
}

#[pymethods]
impl Clipping {
    #[new]
    #[pyo3(signature = (quantile_distribution, p=1.0))]
    fn pynew(py: Python, quantile_distribution: PyObject, p: f32) -> PyResult<Self> {
        let quantile: Box<dyn Samplable<f32>> = extract_distribution(py, quantile_distribution)?;
        Ok(Clipping::new(quantile, p).expect("Error when creating Clipping"))
    }
    #[pyo3(name = "sample_parameters", signature = (audio, rng=None))]
    fn py_sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> ClippingParameters {
        RandomAugmentation::sample_parameters(self, audio, rng)
    }

    #[pyo3(name = "augment")]
    fn py_augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &ClippingParameters,
    ) -> Py<PyArray2<f32>> {
        let augmented =
            RandomAugmentation::augment(self, &waveform.as_array().to_owned(), parameters);
        PyArray2::from_array(py, &augmented).to_owned().into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn create_random_audio(n_samples: usize, sr: usize) -> Audio {
        let mut rng = rand::rng();
        let rand_waveform = Array2::from_shape_fn((1_usize, n_samples as usize), |_| rng.random());
        Audio {
            waveform: rand_waveform,
            sample_rate: sr,
        }
    }

    #[test]
    fn test_clip() {
        const SR: usize = 16_000;
        let random_audio = create_random_audio(3 * SR, SR);
        let clipped = clip(&random_audio.waveform, 0.8);
        let max_random = random_audio
            .waveform
            .fold(f64::NEG_INFINITY, |max, &val| f64::max(max, val as f64));
        let max_clipped = clipped.fold(f64::NEG_INFINITY, |max, &val| f64::max(max, val as f64));
        assert_ne!(max_random, max_clipped)
    }

    #[test]
    fn test_clipping() {
        const SR: usize = 16_000;
        let random_audio = create_random_audio(3 * SR, SR);
        let quantile_distribution = Box::new(Uniform::new(0.8, 0.9));
        let clipping = Clipping::new(quantile_distribution, 1.0);

        let mut rng = RandomNumberGenerator::new(Some(0));
        let parameters = clipping.sample_parameters(&random_audio, Some(&mut rng));
        let clipped = clipping.augment(&random_audio.waveform, &parameters);
        let max_random = random_audio
            .waveform
            .fold(f64::NEG_INFINITY, |max, &val| f64::max(max, val as f64));
        let max_clipped = clipped.fold(f64::NEG_INFINITY, |max, &val| f64::max(max, val as f64));
        assert_ne!(max_random, max_clipped)
    }

    #[test]
    fn test_noclipping() {
        const SR: usize = 16_000;
        let random_audio = create_random_audio(3 * SR, SR);
        let quantile_distribution = Box::new(Uniform::new(0.8, 0.9));
        let clipping = Clipping::new(quantile_distribution, 0.0);

        let mut rng = RandomNumberGenerator::new(Some(0));
        let parameters = clipping.maybe_sample_parameters(&random_audio, Some(&mut rng));
        let clipped = clipping.maybe_augment(&random_audio.waveform, parameters.as_ref());
        let max_random = random_audio
            .waveform
            .fold(f64::NEG_INFINITY, |max, &val| f64::max(max, val as f64));
        let max_clipped = clipped.fold(f64::NEG_INFINITY, |max, &val| f64::max(max, val as f64));
        assert_eq!(max_random, max_clipped)
    }
}
