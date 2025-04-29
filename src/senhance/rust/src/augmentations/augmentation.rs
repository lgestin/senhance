use crate::audio::Audio;
use crate::augmentations::distributions::RandomNumberGenerator;
use log::{debug, error, info, warn};
use ndarray::Array2;
use rand::Rng;
use std::any::Any;

pub trait RandomAugmentation: Any {
    type Parameters;

    fn name(&self) -> &str;
    fn p(&self) -> f32;

    fn sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Self::Parameters;

    fn maybe_sample_parameters(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> Option<Self::Parameters> {
        let sampled_p: f32;
        if let Some(ref mut rng) = rng {
            sampled_p = rng.random();
        } else {
            let mut rng = rand::rng();
            sampled_p = rng.random::<f32>();
        }
        if sampled_p < self.p() {
            Some(self.sample_parameters(audio, rng))
        } else {
            None
        }
    }

    fn augment(&self, waveform: &Array2<f32>, parameters: &Self::Parameters) -> Array2<f32>;

    fn maybe_augment(
        &self,
        waveform: &Array2<f32>,
        parameters: Option<&Self::Parameters>,
    ) -> Array2<f32> {
        if let Some(parameters) = parameters {
            self.augment(waveform, parameters)
        } else {
            waveform.to_owned()
        }
    }
}

pub trait AnyAugmentation: Send + Sync {
    fn sample_parameters_any(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Box<dyn Any + Send + Sync>;
    fn augment_any(
        &self,
        waveform: &Array2<f32>,
        parameters: &dyn Any,
    ) -> Result<Array2<f32>, String>;
}

impl<A: RandomAugmentation + 'static + Send + Sync> AnyAugmentation for A
where
    A::Parameters: 'static + Send + Sync,
{
    fn sample_parameters_any(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Box<dyn Any + Send + Sync> {
        let parameters = self.sample_parameters(audio, rng);
        println!(
            "Created parameters for '{}' with type '{}'",
            self.name(),
            std::any::type_name::<A::Parameters>()
        );
        Box::new(parameters)
    }
    fn augment_any(
        &self,
        waveform: &Array2<f32>,
        parameters: &dyn Any,
    ) -> Result<Array2<f32>, String> {
        // Try direct downcast first (the normal path)
        if let Some(typed_params) = parameters.downcast_ref::<A::Parameters>() {
            return Ok(self.augment(waveform, typed_params));
        }

        // Try if it's a Box<dyn Any> containing our parameters
        if let Some(boxed_any) = parameters.downcast_ref::<Box<dyn Any + Send + Sync>>() {
            if let Some(typed_params) = boxed_any.downcast_ref::<A::Parameters>() {
                return Ok(self.augment(waveform, typed_params));
            }
        }
        let error_msg = format!(
            "Parameter type mismatch for '{}': expected '{}'",
            self.name(),
            std::any::type_name::<A::Parameters>(),
        );
        return Err(error_msg);
    }
}
