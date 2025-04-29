use ndarray::s;
use ndarray::Array2;
use rubato::SincFixedOut;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

pub fn resample(waveform: Array2<f32>, orig_sr: usize, targ_sr: usize) -> Array2<f32> {
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let resampling_ratio = targ_sr as f64 / orig_sr as f64;

    let n_channels = waveform.shape()[0];
    let n_frames = waveform.shape()[1];
    let mut waveform_vec: Vec<Vec<f32>> = Vec::with_capacity(n_channels);

    for i in 0..n_channels {
        let channel = waveform.slice(s![i, ..]).to_vec();
        waveform_vec.push(channel)
    }

    let resampled_vec = if resampling_ratio >= 1.0 {
        let mut resampler =
            SincFixedIn::<f32>::new(resampling_ratio, 2.0, params, 1024, 1).expect("");
        resampler.process(&waveform_vec, None).unwrap()
    } else {
        let mut resampler =
            SincFixedOut::<f32>::new(resampling_ratio, 2.0, params, 1024, 1).expect("");
        resampler.process(&waveform_vec, None).unwrap()
    };

    let n_resampled_frames = (n_frames as f64 * resampling_ratio).ceil() as usize;
    let mut resampled = Array2::<f32>::zeros((n_channels, n_resampled_frames));
    for (i, channel_data) in resampled_vec.iter().enumerate() {
        // The actual output length might be slightly different due to resampling
        let actual_len = channel_data.len();

        // Make sure we don't go out of bounds
        let copy_len = actual_len.min(n_resampled_frames);

        for j in 0..copy_len {
            resampled[[i, j]] = channel_data[j];
        }
    }
    resampled
}
