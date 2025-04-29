use biquad;
use biquad::Biquad;
use biquad::ToHertz;

const Q_BUTTERWORTH_F32: f32 = core::f32::consts::FRAC_1_SQRT_2;

struct BiquadCoefficients {
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
    a0: f32,
    a1: f32,
    a2: f32,
    b1: f32,
    b2: f32,
}

impl BiquadCoefficients {
    fn new(a0: f32, a1: f32, a2: f32, b1: f32, b2: f32) -> Self {
        let x1 = 0.0;
        let x2 = 0.0;
        let y1 = 0.0;
        let y2 = 0.0;
        Self {
            x1,
            x2,
            y1,
            y2,
            a0,
            a1,
            a2,
            b1,
            b2,
        }
    }

    fn lowpass(sample_rate: f32, cutoff_freq: f32, q: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * cutoff_freq / sample_rate;
        let alpha = omega.sin() / (2.0 * q);
        let cos_omega = omega.cos();

        let b0 = (1.0 - cos_omega) / 2.0;
        let b1 = 1.0 - cos_omega;
        let b2 = (1.0 - cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    fn highpass(sample_rate: f32, cutoff_freq: f32, q: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * cutoff_freq / sample_rate;
        let alpha = omega.sin() / (2.0 * q);
        let cos_omega = omega.cos();

        let b0 = (1.0 + cos_omega) / 2.0;
        let b1 = -(1.0 + cos_omega);
        let b2 = (1.0 + cos_omega) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_omega;
        let a2 = 1.0 - alpha;

        Self::new(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
    }

    fn process(&mut self, waveform: &[f32]) -> Vec<f32> {
        if waveform.len() < 3 {
            return waveform.to_vec();
        }

        let mut filtered = Vec::with_capacity(waveform.len());
        filtered.extend_from_slice(&waveform[..2]);

        // for window in waveform.windows(3) {
        //     let filtered_sample = self.a0 * window[2] + self.a1 * window[1] + self.a2 * window[0]
        //         - self.b1 * filtered.last().unwrap()
        //         - self.b2 * filtered.get(filtered.len() - 2).unwrap();
        //     filtered.push(filtered_sample);
        // }
        // filtered

        let mut f: f32;
        for y in waveform.iter() {
            f = self.a0 * y + self.a1 * self.x1 + self.a2 * self.x2
                - self.b1 * self.y1
                - self.b2 * self.y2;
            filtered.push(f);

            self.x2 = self.x1;
            self.x1 = *y;
            self.y2 = self.y1;
            self.y1 = f;
        }
        filtered
    }
}

pub fn lowpass_window(waveform: &[f32], sample_rate: f32, cutoff_freq: f32) -> Vec<f32> {
    let mut filter = BiquadCoefficients::lowpass(sample_rate, cutoff_freq, Q_BUTTERWORTH_F32);
    let filtered = filter.process(waveform);
    filtered
}

pub fn lowpass(waveform: &[f32], f_cutoff: usize, sr: usize) -> Vec<f32> {
    let f_cutoff_hz = f_cutoff.hz();
    let sr_hz = sr.hz();
    let coeffs = biquad::Coefficients::<f32>::from_params(
        biquad::Type::LowPass,
        sr_hz,
        f_cutoff_hz,
        biquad::Q_BUTTERWORTH_F32,
    )
    .expect("couldnt create biquad coefficients");

    let mut biquad_filter = biquad::DirectForm1::<f32>::new(coeffs);

    let mut output = Vec::new();
    for elem in waveform {
        output.push(biquad_filter.run(*elem));
    }
    output
}

pub fn highpass(waveform: &[f32], f_cutoff: usize, sr: usize) -> Vec<f32> {
    let f_cutoff_hz = f_cutoff.hz();
    let sr_hz = (sr / 1000).khz();
    let coeffs = biquad::Coefficients::<f32>::from_params(
        biquad::Type::HighPass,
        sr_hz,
        f_cutoff_hz,
        biquad::Q_BUTTERWORTH_F32,
    )
    .expect("couldnt create biquad coefficients");

    let mut biquad_filter = biquad::DirectForm1::<f32>::new(coeffs);

    let mut output = Vec::new();
    for elem in waveform {
        output.push(biquad_filter.run(*elem));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_output_length() {
    //     let mut coeff = BiquadCoefficients::lowpass(44100.0, 1000.0, 0.707);
    //     let waveform = vec![1.0, -1.0, 1.0, 0.0, 1.0];
    //     let filtered = coeff.process(&waveform);
    //     assert_eq!(waveform.len(), filtered.len());
    // }

    #[test]
    fn test_short_waveforms() {
        let mut coeff = BiquadCoefficients::highpass(44100.0, 1000.0, 0.707);
        let waveform = vec![1.0, 1.0];
        let filtered = coeff.process(&waveform);
        assert_eq!(waveform, filtered);
    }
}
