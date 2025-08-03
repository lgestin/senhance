use crate::audio::Audio;
use crate::augmentations::augmentation::{
    AnyAugmentation, AnyParameters, Augments, RandomAugmentation,
};
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
}

impl Choose {
    pub fn new(
        augmentations: Vec<Box<dyn AnyAugmentation>>,
        choice_distribution: Option<WeightedCategorical>,
    ) -> Self {
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
        }
    }
}

impl Augments for Choose {
    type Parameters = ChooseParameters;
    fn name(&self) -> &str {
        "choose"
    }

    fn sample_parameters(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<ChooseParameters, String> {
        let choice = self.choice_distribution.sample(rng.as_deref_mut());
        let augmentation = &self.augmentations[choice];
        println!("{:?}", augmentation.name_any());
        let choice_parameters = augmentation.sample_parameters_any(audio, rng).unwrap();
        Ok(ChooseParameters {
            choice,
            choice_parameters: choice_parameters,
        })
    }

    fn augment(&self, waveform: &Array2<f32>, parameters: &ChooseParameters) -> Array2<f32> {
        let augmentation = &self.augmentations[parameters.choice];
        augmentation
            .augment_any(waveform, &parameters.choice_parameters)
            .expect("Augmentation Error")
    }
}

#[pyclass(name = "Choose")]
#[derive(Debug)]
pub struct PyChoose {
    choose: RandomAugmentation<Choose>,
}

#[pymethods]
impl PyChoose {
    #[new]
    #[pyo3(signature = (augmentations, choice_distribution=None, p=1.0))]
    fn pynew<'py>(
        py: Python<'py>,
        augmentations: &Bound<'py, PyList>,
        choice_distribution: Option<WeightedCategorical>,
        p: Option<f32>,
    ) -> PyResult<Self> {
        let mut rust_augmentations: Vec<Box<dyn AnyAugmentation>> =
            Vec::with_capacity(augmentations.len());
        for augmentation in augmentations.iter() {
            println!("{:?}", augmentation);
            rust_augmentations.push(extract_augmentation(py, augmentation.into())?);
        }
        let choose = Choose::new(rust_augmentations, choice_distribution);
        let random_choose: RandomAugmentation<Choose>;
        if let Some(p) = p {
            random_choose = RandomAugmentation::new(choose, p).unwrap();
        } else {
            random_choose = RandomAugmentation::new(choose, 1.0).unwrap();
        }
        Ok(PyChoose {
            choose: random_choose,
        })
    }

    #[pyo3(name="sample_parameters", signature = (audio, rng=None))]
    fn py_sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> PyResult<Option<ChooseParameters>> {
        self.choose
            .sample_parameters(audio, rng)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))
    }

    #[pyo3(name = "augment")]
    fn py_augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &ChooseParameters,
    ) -> Py<PyArray2<f32>> {
        println!("choose params {:?}", parameters);
        let augmented = self
            .choose
            .augment(&waveform.as_array().to_owned(), &Some(parameters.clone()));
        PyArray2::from_array(py, &augmented).to_owned().into()
    }
}
