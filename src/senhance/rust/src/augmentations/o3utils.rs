use pyo3::prelude::*;

use crate::augmentations::distributions::{Normal, Samplable, Uniform};
use pyo3::exceptions::PyTypeError;

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
