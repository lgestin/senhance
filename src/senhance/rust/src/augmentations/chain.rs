use crate::audio::Audio;
use crate::augmentations::augmentation::{AnyAugmentation, RandomAugmentation};
use crate::augmentations::distributions::RandomNumberGenerator;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::any::Any;

use super::clipping::Clipping;

#[pyclass]
pub struct ChainParameters {
    chain_parameters: Vec<Box<dyn Any + Send + Sync>>,
}

#[pyclass]
pub struct Chain {
    augmentations: Vec<Box<dyn AnyAugmentation>>,
    #[pyo3(get)]
    p: f32,
}

impl Chain {
    pub fn new(augmentations: Vec<Box<dyn AnyAugmentation>>, p: f32) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be between 0 and 1");
        Chain { augmentations, p }
    }
}

impl RandomAugmentation for Chain {
    type Parameters = ChainParameters;
    fn name(&self) -> &str {
        "choose"
    }
    fn p(&self) -> f32 {
        self.p
    }
    fn sample_parameters(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> ChainParameters {
        let mut chain_parameters: Vec<Box<dyn Any + Send + Sync>> =
            Vec::with_capacity(self.augmentations.len());
        for augmentation in self.augmentations.iter() {
            let augmentation_parameters =
                augmentation.sample_parameters_any(audio, rng.as_deref_mut());
            chain_parameters.push(augmentation_parameters)
        }
        ChainParameters { chain_parameters }
    }
    fn augment(&self, waveform: &Array2<f32>, parameters: &ChainParameters) -> Array2<f32> {
        let mut augmented_waveform: Array2<f32> = waveform.clone();
        for (augmentation, parameters) in self
            .augmentations
            .iter()
            .zip(parameters.chain_parameters.iter())
        {
            augmented_waveform = augmentation
                .augment_any(&augmented_waveform, parameters)
                .expect("Augmentation Error")
        }
        augmented_waveform
    }
}

#[pymethods]
impl Chain {
    #[new]
    #[pyo3(signature = (augmentations, p=1.0))]
    fn pynew<'py>(augmentations: &Bound<'py, PyList>, p: f32) -> PyResult<Self> {
        let mut aug: Vec<Box<dyn AnyAugmentation>> = Vec::with_capacity(augmentations.len());
        for item in augmentations.iter() {
            if let Ok(augmentation) = item.extract::<Clipping>() {
                aug.push(Box::new(augmentation));
            }
        }
        Ok(Chain::new(aug, p))
    }
    #[pyo3(signature = (audio, rng=None))]
    fn sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> ChainParameters {
        RandomAugmentation::sample_parameters(self, audio, rng)
    }
    fn augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &ChainParameters,
    ) -> Py<PyArray2<f32>> {
        let augmented =
            RandomAugmentation::augment(self, &waveform.as_array().to_owned(), parameters);
        PyArray2::from_array(py, &augmented).to_owned().into()
    }
}
