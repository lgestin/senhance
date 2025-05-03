use crate::audio::Audio;
use crate::augmentations::augmentation::{AnyAugmentation, AnyParameters, RandomAugmentation};
use crate::augmentations::distributions::{RandomNumberGenerator, Samplable, WeightedCategorical};
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::o3utils::extract_augmentation;

#[pyclass]
#[derive(Debug, Clone)]
pub struct ChooseParameters {
    #[pyo3(get)]
    choice: usize,
    choice_parameters: AnyParameters,
}

#[pyclass]
#[derive(Debug, Clone)]
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
        let augmentation = &self.augmentations[choice];
        println!("{:?}", augmentation.name_any());
        let choice_parameters = augmentation.sample_parameters_any(audio, rng);
        ChooseParameters {
            choice,
            choice_parameters: choice_parameters,
        }
    }

    fn augment(&self, waveform: &Array2<f32>, parameters: &ChooseParameters) -> Array2<f32> {
        let augmentation = &self.augmentations[parameters.choice];
        augmentation
            .augment_any(waveform, &parameters.choice_parameters)
            .expect("Augmentation Error")
    }
}

#[pymethods]
impl Choose {
    #[new]
    #[pyo3(signature = (augmentations, choice_distribution=None, p=1.0))]
    fn pynew<'py>(
        py: Python<'py>,
        augmentations: &Bound<'py, PyList>,
        choice_distribution: Option<WeightedCategorical>,
        p: f32,
    ) -> PyResult<Self> {
        let mut rust_augmentations: Vec<Box<dyn AnyAugmentation>> =
            Vec::with_capacity(augmentations.len());
        for augmentation in augmentations.iter() {
            println!("{:?}", augmentation);
            rust_augmentations.push(extract_augmentation(py, augmentation.into())?);
        }
        Ok(Choose::new(rust_augmentations, choice_distribution, p))
    }

    #[pyo3(name="sample_parameters", signature = (audio, rng=None))]
    fn py_sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> ChooseParameters {
        self.sample_parameters(audio, rng)
    }

    #[pyo3(name = "augment")]
    fn py_augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &ChooseParameters,
    ) -> Py<PyArray2<f32>> {
        println!("choose params {:?}", parameters);
        let augmented = self.augment(&waveform.as_array().to_owned(), parameters);
        PyArray2::from_array(py, &augmented).to_owned().into()
    }
}
