use crate::audio::Audio;
use crate::augmentations::augmentation::{AnyAugmentation, RandomAugmentation};
use crate::augmentations::distributions::{RandomNumberGenerator, Samplable, WeightedCategorical};
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use std::any::Any;

use super::clipping::Clipping;

#[pyclass]
#[derive(Debug)]
pub struct ChooseParameters {
    #[pyo3(get)]
    choice: usize,
    choice_parameters: Box<dyn Any + Send + Sync>,
}

#[pyclass]
pub struct Choose {
    augmentations: Vec<Box<dyn AnyAugmentation>>,
    #[pyo3(get)]
    choice_distribution: WeightedCategorical,
    #[pyo3(get)]
    p: f32,
}

impl Choose {
    pub fn new(
        augmentations: Vec<Box<dyn AnyAugmentation>>,
        choice_distribution: Option<WeightedCategorical>,
        p: f32,
    ) -> Self {
        assert!((0.0..=1.0).contains(&p), "p must be between 0 and 1");
        let weights = if let Some(choice_distribution) = choice_distribution {
            choice_distribution
        } else {
            let n = augmentations.len();
            let choice_weights = vec![1.0 / n as f32; n];
            WeightedCategorical::new(choice_weights)
        };

        Choose {
            augmentations,
            choice_distribution: weights,
            p,
        }
    }
}

impl RandomAugmentation for Choose {
    type Parameters = ChooseParameters;
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
    ) -> ChooseParameters {
        let choice = self.choice_distribution.sample(rng.as_deref_mut());
        let choice_parameters = self.augmentations[choice].sample_parameters_any(audio, rng);
        ChooseParameters {
            choice,
            choice_parameters: Box::new(choice_parameters),
        }
    }
    fn augment(&self, waveform: &Array2<f32>, parameters: &ChooseParameters) -> Array2<f32> {
        let augmentation = &self.augmentations[parameters.choice];
        augmentation
            .augment_any(waveform, parameters.choice_parameters.as_ref())
            .expect("Augmentation Error")
    }
}

#[pymethods]
impl Choose {
    #[new]
    #[pyo3(signature = (augmentations, choice_distribution=None, p=1.0))]
    fn pynew<'py>(
        augmentations: &Bound<'py, PyList>,
        choice_distribution: Option<WeightedCategorical>,
        p: f32,
    ) -> PyResult<Self> {
        let mut aug: Vec<Box<dyn AnyAugmentation>> = Vec::with_capacity(augmentations.len());
        for item in augmentations.iter() {
            println!("{:?}", item);
            if let Ok(augmentation) = item.extract::<Clipping>() {
                aug.push(Box::new(augmentation));
            }
        }
        Ok(Choose::new(aug, choice_distribution, p))
    }
    #[pyo3(signature = (audio, rng=None))]
    fn sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> ChooseParameters {
        RandomAugmentation::sample_parameters(self, audio, rng)
    }
    fn augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &ChooseParameters,
    ) -> Py<PyArray2<f32>> {
        println!("{:?}", parameters);
        let augmented =
            RandomAugmentation::augment(self, &waveform.as_array().to_owned(), parameters);
        PyArray2::from_array(py, &augmented).to_owned().into()
    }
}
