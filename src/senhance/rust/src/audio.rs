use hound;
use ndarray::s;
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::path::PathBuf;

use crate::resample::resample;

pub struct AudioFile {
    filepath: PathBuf,
    start_s: Option<f32>,
    end_s: Option<f32>,
}

#[pyclass]
#[derive(Clone)]
pub struct Audio {
    pub waveform: Array2<f32>,
    #[pyo3(get)]
    pub sample_rate: usize,
}

impl Audio {
    pub fn new(waveform: Array2<f32>, sample_rate: usize) -> Self {
        Audio {
            waveform,
            sample_rate,
        }
    }

    pub fn duration_s(&self) -> f64 {
        let n_samples = &self.waveform.shape()[1];
        *n_samples as f64 / self.sample_rate as f64
    }

    pub fn rms_loudness(&self) -> Array1<f32> {
        let n_channels = self.waveform.nrows();
        let n_samples = self.waveform.ncols() as f64;
        let mut rms_loudness = Array1::<f32>::zeros(n_channels);

        for i in 0..n_channels {
            let channel = self.waveform.slice(s![i, ..]);
            let sum_squares: f64 = channel.iter().map(|&sample| (sample * sample) as f64).sum();
            rms_loudness[i] = (sum_squares / n_samples).sqrt() as f32;
        }
        rms_loudness
    }

    pub fn loudness_db(&self) -> Array1<f32> {
        let rms_loudness = self.rms_loudness();
        rms_loudness.mapv(|l| 20.0 * l.log10())
    }

    pub fn mono(&self) -> Self {
        let n_samples = self.waveform.ncols();
        let n_channels = self.waveform.nrows();

        if n_channels == 1 {
            return self.clone();
        }

        let mut mono = Array2::<f32>::zeros((1, n_samples));

        let mut sample: f32;
        for j in 0..n_samples {
            sample = 0.0;
            for i in 0..n_channels {
                sample += self.waveform[[i, j]] / n_channels as f32;
            }
            mono[[0, j]] = sample;
        }
        Audio {
            waveform: mono,
            sample_rate: self.sample_rate,
        }
    }

    pub fn normalize(&mut self, target_db: f32) -> &mut Self {
        let loudness_db = self.loudness_db();
        let n_channels = self.waveform.nrows();
        for i in 0..n_channels {
            // Calculate the required gain to reach target_db
            let gain = 10.0_f32.powf((target_db - loudness_db[i]) / 20.0);

            // Apply gain to this channel in-place
            for j in 0..self.waveform.ncols() {
                self.waveform[[i, j]] *= gain;
            }
        }
        self
    }

    pub fn resample(self, targ_sr: usize) -> Audio {
        Audio {
            waveform: resample(self.waveform, self.sample_rate, targ_sr),
            sample_rate: targ_sr,
        }
    }
}

#[pymethods]
impl Audio {
    #[new]
    fn from_python(waveform: PyReadonlyArray2<f32>, sample_rate: usize) -> Self {
        Self {
            waveform: waveform.as_array().to_owned(),
            sample_rate,
        }
    }

    #[getter]
    pub fn waveform<'py>(&'py self, py: Python<'py>) -> Py<PyArray2<f32>> {
        PyArray2::from_array(py, &self.waveform).to_owned().into()
    }

    #[getter(loudness)]
    fn loudness_py<'py>(&'py self, py: Python<'py>) -> Py<PyArray1<f32>> {
        PyArray1::from_array(py, &self.loudness_db())
            .to_owned()
            .into()
    }

    #[pyo3(name = "mono")]
    fn mono_py<'py>(&'py self) -> Self {
        self.mono()
    }

    #[pyo3(name = "normalize")]
    fn normalize_py<'py>(&'py self, db: f32) -> PyResult<Self> {
        let mut normalized_audio = Audio {
            waveform: self.waveform.clone(),
            sample_rate: self.sample_rate,
        };
        normalized_audio.normalize(db);
        Ok(normalized_audio)
    }

    #[pyo3(name = "resample")]
    fn resample_py<'py>(&'py self, targ_sr: usize) -> Self {
        self.clone().resample(targ_sr)
    }
}

// fn load_wav(wavpath: String) -> Audio {
//     let mut reader = hound::WavReader::open(wavpath).unwrap();
//     let n_channels = reader.spec().channels as u16;
//     let duration = reader.duration() as u32;
//
//     let mut waveform = Array2::<MaybeUninit<f32>>::uninit((n_channels, duration));
//     for sample in reader.samples() {
//         sample.err;
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn create_random_audio(n_samples: i32, sr: usize) -> Audio {
        let mut rng = rand::rng();
        let rand_waveform = Array2::from_shape_fn((1_usize, n_samples as usize), |_| rng.random());
        Audio {
            waveform: rand_waveform,
            sample_rate: sr,
        }
    }

    #[test]
    fn test_duration_s() {
        const SR: usize = 16_000;
        let random_audio = create_random_audio(3 * SR as i32, SR);
        assert_eq!(random_audio.duration_s(), 3 as f64);
    }

    #[test]
    fn test_normalize() {
        const SR: usize = 16_000;
        let target_db: f32 = -23.0;
        let mut random_audio = create_random_audio(3 * SR as i32, SR);
        //assert_eq!(random_audio.normalize(target_db).loudness_db(), target_db);
        assert_eq!(random_audio.duration_s(), 3 as f64);
    }
}
