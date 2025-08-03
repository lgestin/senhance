// use hound;
use ndarray::s;
use ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::fmt;
use std::path::Path;

use crate::resample::resample;

#[pyclass(str = "Audio(waveform, sample_rate={sample_rate:?})")]
#[derive(Clone)]
pub struct Audio {
    pub waveform: Array2<f32>,
    #[pyo3(get)]
    pub sample_rate: Option<usize>,
}

fn load_wav<P: AsRef<Path>>(
    path: P,
    offset_s: Option<f64>,
    duration_s: Option<f64>,
) -> Result<Audio, hound::Error> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate as usize;
    let n_channels = spec.channels as usize;
    let total_samples = reader.len() as usize;

    let offset: usize;
    if let Some(offset_s) = offset_s {
        offset = (offset_s * (sample_rate as f64)) as usize;
    } else {
        offset = 0;
    }

    let n_samples: usize;
    if let Some(duration_s) = duration_s {
        n_samples = (duration_s * (sample_rate as f64)) as usize;
    } else {
        n_samples = total_samples - offset;
    }

    let mut waveform = Array2::<f32>::zeros((n_channels, n_samples));

    let mut sample_idx = 0;
    reader.seek(offset as u32).unwrap();
    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut samples_iter = reader.samples::<f32>();
            while sample_idx < n_samples {
                let channel = sample_idx % n_channels;
                match samples_iter.next() {
                    Some(Ok(sample)) => waveform[[channel, sample_idx]] = sample,
                    Some(Err(e)) => return Err(e),
                    None => break,
                }
                sample_idx += 1;
            }
        }
        hound::SampleFormat::Int => {
            let bit_depth = spec.bits_per_sample;
            let max_val = (1i32 << (bit_depth - 1)) as f32;

            let mut samples_iter = reader.samples::<i32>();
            while sample_idx < n_samples {
                let channel = sample_idx % n_channels;
                match samples_iter.next() {
                    Some(Ok(sample)) => waveform[[channel, sample_idx]] = sample as f32 / max_val,
                    Some(Err(e)) => return Err(e),
                    None => break,
                }
                sample_idx += 1;
            }
        }
    }
    Ok(Audio::new(waveform, Some(sample_rate)))
}

impl Audio {
    pub fn new(waveform: Array2<f32>, sample_rate: Option<usize>) -> Self {
        Audio {
            waveform,
            sample_rate,
        }
    }

    pub fn from_wav<P: AsRef<Path>>(
        path: P,
        offset_s: Option<f64>,
        duration_s: Option<f64>,
    ) -> Result<Self, hound::Error> {
        let path = path.as_ref();
        if !(path.is_file()) {
            return Err(hound::Error::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("File {} doesn't exist", path.display()),
            )));
        }

        if let Some(ext) = path.extension() {
            if ext.to_string_lossy().to_lowercase() == "wav" {
                let audio = load_wav(path, offset_s, duration_s)?;
                return Ok(audio);
            }
        }
        Err(hound::Error::FormatError("Unsupported file format"))
    }

    fn require_sample_rate(&self) -> Result<usize, String> {
        self.sample_rate
            .ok_or_else(|| "sample_rate needs to be defined for this operation".to_string())
    }

    pub fn duration_s(&self) -> Result<f64, String> {
        let sample_rate = self.require_sample_rate()?;
        let n_samples = &self.waveform.shape()[1];
        Ok(*n_samples as f64 / sample_rate as f64)
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

    pub fn resample(self, targ_sr: usize) -> Result<Audio, String> {
        let sample_rate = self.require_sample_rate()?;
        Ok(Audio {
            waveform: resample(self.waveform, sample_rate, targ_sr),
            sample_rate: Some(targ_sr),
        })
    }
}

impl fmt::Display for Audio {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(sample_rate) = self.sample_rate {
            write!(f, "Audio(waveform, sample_rate={})", sample_rate)
        } else {
            write!(f, "Audio(waveform, sample_rate=None)")
        }
    }
}

#[pymethods]
impl Audio {
    #[new]
    #[pyo3(signature = (waveform, sample_rate=None))]
    fn from_python(waveform: PyReadonlyArray2<f32>, sample_rate: Option<usize>) -> Self {
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
        self.clone().resample(targ_sr).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn create_random_audio(n_samples: i32, sr: usize) -> Audio {
        let mut rng = rand::rng();
        let rand_waveform = Array2::from_shape_fn((1_usize, n_samples as usize), |_| rng.random());
        Audio {
            waveform: rand_waveform,
            sample_rate: Some(sr),
        }
    }

    #[test]
    fn test_load_audio_from_file() {
        let path = Path::new("/home/lucas/code/senhance/tests/assets/physicsworks.wav");
        let audio = Audio::from_wav(path, None, None).unwrap();
        assert_eq!(audio.waveform.shape(), &[1, 3252535]);
        let audio = Audio::from_wav(path, Some(1.0), None).unwrap();
        assert_eq!(audio.waveform.shape(), &[1, 3236535]);
        let audio = Audio::from_wav(path, None, Some(1.0)).unwrap();
        assert_eq!(audio.waveform.shape(), &[1, 16000]);
        let audio = Audio::from_wav(path, Some(1.0), Some(1.0)).unwrap();
        assert_eq!(audio.waveform.shape(), &[1, 16000]);
    }

    #[test]
    fn test_duration_s() {
        const SR: usize = 16_000;
        let random_audio = create_random_audio(3 * SR as i32, SR);
        assert_eq!(random_audio.duration_s().unwrap(), 3 as f64);
    }

    #[test]
    fn test_normalize() {
        const SR: usize = 16_000;
        let target_db: f32 = -23.0;
        let mut random_audio = create_random_audio(3 * SR as i32, SR);
        let normalized_db: f32 = random_audio.normalize(target_db).loudness_db()[0];
        assert!(
            (normalized_db - target_db).abs() < 0.001,
            "Expected loudness to be approximately {}, but got {}",
            target_db,
            normalized_db
        );
        assert_eq!(random_audio.duration_s().unwrap(), 3 as f64);
    }
}
