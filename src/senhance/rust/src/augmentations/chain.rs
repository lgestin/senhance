use crate::audio::Audio;
use crate::augmentations::augmentation::{
    AnyAugmentation, AnyParameters, Augments, RandomAugmentation,
};
use crate::augmentations::distributions::RandomNumberGenerator;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::o3utils::extract_augmentation;

#[derive(Debug, Clone)]
pub struct ChainParameters<T> {
    chain_parameters: Vec<T>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct Chain {
    augmentations: Vec<Box<dyn AnyAugmentation>>,
}

impl Chain {
    pub fn new(augmentations: Vec<Box<dyn AnyAugmentation>>) -> Self {
        Self { augmentations }
    }
}

impl Augments for Chain {
    type Parameters = ChainParameters<Option<AnyParameters>>;

    fn name(&self) -> &str {
        "choose"
    }

    fn sample_parameters(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<ChainParameters<Option<AnyParameters>>, String> {
        let mut chain_parameters: Vec<Option<AnyParameters>> =
            Vec::with_capacity(self.augmentations.len());
        println!("n augmentations {:?}", self.augmentations.len());
        for augmentation in self.augmentations.iter() {
            println!("{:?}", augmentation.name_any());
            let augmentation_parameters = augmentation
                .sample_parameters_any(audio, rng.as_deref_mut())
                .unwrap();
            chain_parameters.push(Some(augmentation_parameters))
        }
        Ok(ChainParameters { chain_parameters })
    }

    fn augment(
        &self,
        waveform: &Array2<f32>,
        parameters: &ChainParameters<Option<AnyParameters>>,
    ) -> Array2<f32> {
        let mut augmented_waveform: Array2<f32> = waveform.clone();
        for (augmentation, parameters) in self
            .augmentations
            .iter()
            .zip(parameters.chain_parameters.iter())
        {
            if let Some(parameters) = parameters {
                augmented_waveform = augmentation
                    .augment_any(&augmented_waveform, &parameters)
                    .expect("Augmentation Error")
            }
        }
        augmented_waveform
    }
}

#[pyclass(name = "ChainParameters")]
#[derive(Debug, Clone)]
pub struct PyChainParameters {
    chain_parameters: Vec<Option<AnyParameters>>,
}

impl PyChainParameters {
    fn from_rust(rust_chain_parameters: ChainParameters<Option<AnyParameters>>) -> Self {
        Self {
            chain_parameters: rust_chain_parameters.chain_parameters,
        }
    }
    fn to_rust(&self) -> ChainParameters<Option<AnyParameters>> {
        let chain_parameters = self.chain_parameters.clone();
        ChainParameters {
            chain_parameters: chain_parameters,
        }
    }
}

#[pyclass(name = "Chain")]
#[derive(Debug)]
pub struct PyChain {
    chain: RandomAugmentation<Chain>,
}

#[pymethods]
impl PyChain {
    #[new]
    #[pyo3(signature = (augmentations, p=1.0))]
    fn pynew<'py>(
        py: Python<'py>,
        augmentations: &Bound<'py, PyList>,
        p: Option<f32>,
    ) -> PyResult<Self> {
        let mut rust_augmentations: Vec<Box<dyn AnyAugmentation>> =
            Vec::with_capacity(augmentations.len());
        for augmentation in augmentations.iter() {
            rust_augmentations.push(extract_augmentation(py, augmentation.into())?);
        }
        let chain = Chain::new(rust_augmentations);
        let random_chain: RandomAugmentation<Chain>;
        if let Some(p) = p {
            random_chain = RandomAugmentation::new(chain, p).unwrap();
        } else {
            random_chain = RandomAugmentation::new(chain, 1.0).unwrap();
        }
        Ok(PyChain {
            chain: random_chain,
        })
    }

    #[pyo3(name = "sample_parameters", signature = (audio, rng=None))]
    fn py_sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> PyResult<Option<PyChainParameters>> {
        let rust_chain_parameters = self
            .chain
            .sample_parameters(audio, rng)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;
        if let Some(parameters) = rust_chain_parameters {
            Ok(Some(PyChainParameters::from_rust(parameters)))
        } else {
            Ok(None)
        }
    }

    #[pyo3(name = "augment")]
    fn py_augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &PyChainParameters,
    ) -> Py<PyArray2<f32>> {
        let rust_parameters = parameters.to_rust();
        let augmented = self.chain.augment(
            &waveform.as_array().to_owned(),
            &Some(rust_parameters.clone()),
        );
        PyArray2::from_array(py, &augmented).to_owned().into()
    }

    // fn __getitem__<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<PyObject> {
    //     if idx >= self.augmentations.len() {
    //         return Err(pyo3::exceptions::PyIndexError::new_err(format!(
    //             "Index {} out of range for Chain with length {}",
    //             idx,
    //             self.augmentations.len()
    //         )));
    //     }
    //     return Ok(self.augmentations[idx].to_py(py));
    // }
}
