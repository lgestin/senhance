use pyo3::prelude::*;

use crate::augmentations::distributions::{Normal, Samplable, Uniform};
use pyo3::exceptions::PyTypeError;

use super::augmentation::AnyAugmentation;
use super::chain::PyChain;
use super::choose::PyChoose;
use super::clipping::PyClipping;
use super::random_noise::PyRandomNoise;

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
    if let Ok(augmentation) = pyaugmentation.extract::<PyChoose>(py) {
        Ok(Box::new(augmentation.choose.augmentation))
    } else if let Ok(augmentation) = pyaugmentation.extract::<PyChain>(py) {
        Ok(Box::new(augmentation.chain.augmentation))
    } else if let Ok(augmentation) = pyaugmentation.extract::<PyClipping>(py) {
        Ok(Box::new(augmentation.clipping.augmentation))
    } else if let Ok(augmentation) = pyaugmentation.extract::<PyRandomNoise>(py) {
        Ok(Box::new(augmentation.random_noise.augmentation))
    } else {
        Err(PyErr::new::<PyTypeError, _>(
            "quantile_distribution should either be Uniform, Normal or TruncatedNormal",
        ))
    }
}
