mod audio;
mod augmentations;
mod filter;
mod resample;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[pyo3(name = "resample")]
fn resample_py<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray2<f32>,
    orig_sr: usize,
    targ_sr: usize,
) -> PyResult<Py<PyArray2<f32>>> {
    let audio_vec = audio.as_array();
    let resampled: Array2<f32> = resample::resample(audio_vec.to_owned(), orig_sr, targ_sr);
    Ok(PyArray2::from_array(py, &resampled).to_owned().into())
}

#[pyfunction]
#[pyo3(name = "lowpass")]
fn lowpass_py<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<f32>,
    f_cutoff: usize,
    sr: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let audio_vec = audio.as_slice()?;
    let resampled_vec: Vec<f32> = filter::lowpass(audio_vec, f_cutoff, sr);
    Ok(resampled_vec.into_pyarray(py).to_owned().into())
}

#[pyfunction]
#[pyo3(name = "lowpass_window")]
fn lowpass_window_py<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<f32>,
    f_cutoff: f32,
    sr: f32,
) -> PyResult<Py<PyArray1<f32>>> {
    let audio_vec = audio.as_slice()?;
    let resampled_vec: Vec<f32> = filter::lowpass_window(audio_vec, sr, f_cutoff);
    Ok(resampled_vec.into_pyarray(py).to_owned().into())
}

#[pyfunction]
#[pyo3(name = "highpass")]
fn highpass_py<'py>(
    py: Python<'py>,
    audio: PyReadonlyArray1<f32>,
    f_cutoff: usize,
    sr: usize,
) -> PyResult<Py<PyArray1<f32>>> {
    let audio_vec = audio.as_slice()?;
    let resampled_vec: Vec<f32> = filter::highpass(audio_vec, f_cutoff, sr);
    Ok(resampled_vec.into_pyarray(py).to_owned().into())
}

#[pymodule]
fn rust_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<audio::Audio>()?;
    m.add_function(wrap_pyfunction!(resample_py, m)?)?;
    m.add_function(wrap_pyfunction!(lowpass_py, m)?)?;
    m.add_function(wrap_pyfunction!(lowpass_window_py, m)?)?;
    m.add_function(wrap_pyfunction!(highpass_py, m)?)?;
    m.add_class::<augmentations::distributions::Uniform>()?;
    m.add_class::<augmentations::distributions::WeightedCategorical>()?;
    m.add_class::<augmentations::distributions::RandomNumberGenerator>()?;
    m.add_class::<augmentations::clipping::Clipping>()?;
    m.add_class::<augmentations::random_noise::RandomNoise>()?;
    m.add_class::<augmentations::chain::Chain>()?;
    m.add_class::<augmentations::choose::Choose>()?;
    Ok(())
}
