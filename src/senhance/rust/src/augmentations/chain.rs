use crate::audio::Audio;
use crate::augmentations::augmentation::{
    AnyAugmentation, AnyParameters, RandomAugmentation, SampledParameters,
};
use crate::augmentations::distributions::RandomNumberGenerator;
use ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;

use super::choose::Choose;
use super::clipping::Clipping;
use super::random_noise::RandomNoise;

#[derive(Debug, Clone)]
pub struct ChainParameters<T> {
    chain_parameters: Vec<T>,
}

#[pyclass]
#[derive(Debug, Clone)]
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
    type Parameters = ChainParameters<SampledParameters<AnyParameters>>;

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
    ) -> ChainParameters<SampledParameters<AnyParameters>> {
        let mut chain_parameters: Vec<SampledParameters<AnyParameters>> =
            Vec::with_capacity(self.augmentations.len());
        println!("n augmentations {:?}", self.augmentations.len());
        for augmentation in self.augmentations.iter() {
            println!("{:?}", augmentation.name_any());
            let augmentation_parameters =
                augmentation.sample_parameters_any(audio, rng.as_deref_mut());
            chain_parameters.push(SampledParameters::Sampled(augmentation_parameters))
        }
        ChainParameters { chain_parameters }
    }

    fn maybe_sample_parameters(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> SampledParameters<ChainParameters<SampledParameters<AnyParameters>>> {
        let sampled_p: f32 = if let Some(rng) = rng.as_deref_mut() {
            rng.rand()
        } else {
            RandomNumberGenerator::new(None).rand()
        };

        if sampled_p >= self.p() {
            return SampledParameters::NotSampled;
        }

        let mut chain_parameters: Vec<SampledParameters<AnyParameters>> =
            Vec::with_capacity(self.augmentations.len());
        for augmentation in self.augmentations.iter() {
            let augmentation_parameters =
                augmentation.maybe_sample_parameters_any(audio, rng.as_deref_mut());
            chain_parameters.push(augmentation_parameters)
        }
        SampledParameters::Sampled(ChainParameters { chain_parameters })
    }

    fn augment(
        &self,
        waveform: &Array2<f32>,
        parameters: &ChainParameters<SampledParameters<AnyParameters>>,
    ) -> Array2<f32> {
        let mut augmented_waveform: Array2<f32> = waveform.clone();
        for (augmentation, parameters) in self
            .augmentations
            .iter()
            .zip(parameters.chain_parameters.iter())
        {
            if let SampledParameters::Sampled(parameters) = parameters {
                augmented_waveform = augmentation
                    .augment_any(&augmented_waveform, &parameters)
                    .expect("Augmentation Error")
            }
        }
        augmented_waveform
    }

    fn maybe_augment(
        &self,
        waveform: &Array2<f32>,
        parameters: &SampledParameters<ChainParameters<SampledParameters<AnyParameters>>>,
    ) -> Result<Array2<f32>, String> {
        match parameters {
            SampledParameters::NotSampled => Ok(waveform.to_owned()),
            SampledParameters::Sampled(parameters) => {
                let mut augmented_waveform: Array2<f32> = waveform.clone();
                for (augmentation, augmentation_parameters) in self
                    .augmentations
                    .iter()
                    .zip(parameters.chain_parameters.iter())
                {
                    if let SampledParameters::Sampled(augmentation_parameters) =
                        augmentation_parameters
                    {
                        augmented_waveform = augmentation
                            .maybe_augment_any(
                                &augmented_waveform,
                                SampledParameters::Sampled(&augmentation_parameters),
                            )
                            .expect("Augmentation Error")
                    }
                }
                Ok(augmented_waveform)
            }
            SampledParameters::None => Err("se".to_string()),
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct PyChainParameters {
    chain_parameters: Vec<SampledParameters<AnyParameters>>,
}

impl PyChainParameters {
    fn from_rust(rust_chain_parameters: ChainParameters<SampledParameters<AnyParameters>>) -> Self {
        Self {
            chain_parameters: rust_chain_parameters.chain_parameters,
        }
    }
    fn to_rust(&self) -> ChainParameters<SampledParameters<AnyParameters>> {
        let chain_parameters = self.chain_parameters.clone();
        ChainParameters {
            chain_parameters: chain_parameters,
        }
    }
}

#[pymethods]
impl Chain {
    #[new]
    #[pyo3(signature = (augmentations, p=1.0))]
    fn pynew<'py>(augmentations: &Bound<'py, PyList>, p: f32) -> PyResult<Self> {
        let mut rust_augmentations: Vec<Box<dyn AnyAugmentation>> =
            Vec::with_capacity(augmentations.len());
        for augmentation in augmentations.iter() {
            if let Ok(rust_augmentation) = augmentation.extract::<Choose>() {
                rust_augmentations.push(Box::new(rust_augmentation));
            } else if let Ok(rust_augmentation) = augmentation.extract::<Chain>() {
                rust_augmentations.push(Box::new(rust_augmentation));
            } else if let Ok(rust_augmentation) = augmentation.extract::<Clipping>() {
                rust_augmentations.push(Box::new(rust_augmentation));
            } else if let Ok(rust_augmentation) = augmentation.extract::<RandomNoise>() {
                rust_augmentations.push(Box::new(rust_augmentation));
            }
        }
        Ok(Chain::new(rust_augmentations, p))
    }

    #[pyo3(name = "sample_parameters", signature = (audio, rng=None))]
    fn py_sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> PyChainParameters {
        let rust_chain_parameters = self.sample_parameters(audio, rng);
        PyChainParameters::from_rust(rust_chain_parameters)
    }

    #[pyo3(name = "augment")]
    fn py_augment<'py>(
        &'py self,
        py: Python<'py>,
        waveform: PyReadonlyArray2<f32>,
        parameters: &PyChainParameters,
    ) -> Py<PyArray2<f32>> {
        let rust_parameters = parameters.to_rust();
        let augmented = self.augment(&waveform.as_array().to_owned(), &rust_parameters);
        PyArray2::from_array(py, &augmented).to_owned().into()
    }

    //fn __getitem__<'py>(&self, py: Python<'py>, idx: usize) -> PyResult<PyObject> {
    //    if idx >= self.augmentations.len() {
    //        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
    //            "Index {} out of range for Chain with length {}",
    //            idx,
    //            self.augmentations.len()
    //        )));
    //    }
    //
    //    // Check the concrete type of the augmentation
    //    if let Some(clipping) = self.augmentations[idx].downcast_ref::<Clipping>() {
    //        // Clone the Clipping and convert to Python object
    //        Ok(Py::new(py, clipping.clone())?.into())
    //    } else if let Some(random_noise) = self.augmentations[idx].downcast_ref::<RandomNoise>() {
    //        // Clone the RandomNoise and convert to Python object
    //        Ok(Py::new(py, random_noise.clone())?.into())
    //    } else {
    //        // Handle case where augmentation is neither Clipping nor RandomNoise
    //        Err(pyo3::exceptions::PyRuntimeError::new_err(
    //            "Unsupported augmentation type in Chain",
    //        ))
    //    }
    //}
}
