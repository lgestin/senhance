use pyo3::prelude::*;

use crate::augmentations::distributions::{Normal, Samplable, Uniform};
use pyo3::exceptions::PyTypeError;

use super::augmentation::AnyAugmentation;
use super::chain::Chain;
use super::choose::Choose;
use super::clipping::Clipping;
use super::random_noise::RandomNoise;

pub fn extract_distribution<'py>(
    py: Python<'py>,
    pydistribution: PyObject,
) -> PyResult<Box<dyn Samplable<f32>>> {
    if let Ok(distribution) = pydistribution.extract::<Uniform>(py) {
        Ok(Box::new(distribution))
    } else if let Ok(distribution) = pydistribution.extract::<Normal>(py) {
        Ok(Box::new(distribution))
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "quantile_distribution should either be Uniform, Normal or TruncatedNormal",
        ))
    }
}

pub fn extract_augmentation<'py>(
    py: Python<'py>,
    pyaugmentation: PyObject,
) -> PyResult<Box<dyn AnyAugmentation>> {
    if let Ok(augmentation) = pyaugmentation.extract::<Choose>(py) {
        Ok(Box::new(augmentation))
    } else if let Ok(augmentation) = pyaugmentation.extract::<Chain>(py) {
        Ok(Box::new(augmentation))
    } else if let Ok(augmentation) = pyaugmentation.extract::<Clipping>(py) {
        Ok(Box::new(augmentation))
    } else if let Ok(augmentation) = pyaugmentation.extract::<RandomNoise>(py) {
        Ok(Box::new(augmentation))
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "quantile_distribution should either be Uniform, Normal or TruncatedNormal",
        ))
    }
}
