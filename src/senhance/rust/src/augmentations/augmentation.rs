use crate::audio::Audio;
use crate::augmentations::distributions::RandomNumberGenerator;
use ndarray::Array2;
use std::any::Any;
use std::fmt::{Debug, Formatter, Result as FmtResult};

#[derive(Debug, Clone)]
pub enum SampledParameters<P> {
    Sampled(P),
    NotSampled,
    None,
}

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
    ) -> SampledParameters<Self::Parameters> {
        let sampled_p: f32 = if let Some(rng) = rng.as_deref_mut() {
            rng.rand()
        } else {
            RandomNumberGenerator::new(None).rand()
        };

        if sampled_p < self.p() {
            SampledParameters::Sampled(self.sample_parameters(audio, rng))
        } else {
            SampledParameters::NotSampled
        }
    }

    fn augment(&self, waveform: &Array2<f32>, parameters: &Self::Parameters) -> Array2<f32>;

    fn maybe_augment(
        &self,
        waveform: &Array2<f32>,
        parameters: &SampledParameters<Self::Parameters>,
    ) -> Result<Array2<f32>, String> {
        match parameters {
            SampledParameters::Sampled(parameters) => Ok(self.augment(waveform, parameters)),
            SampledParameters::NotSampled => Ok(waveform.to_owned()),
            SampledParameters::None => {
                let audio = Audio::new(waveform.clone(), 0);
                let sampled_parameters = self.sample_parameters(&audio, None);
                Ok(self.augment(waveform, &sampled_parameters))
            }
        }
    }
}

pub trait CloneableAny: Any + Send + Sync {
    fn name(&self) -> &'static str;
    fn clone_box(&self) -> Box<dyn CloneableAny>;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Clone + Any + Send + Sync + 'static> CloneableAny for T {
    fn name(&self) -> &'static str {
        std::any::type_name::<T>()
    }
    fn clone_box(&self) -> Box<dyn CloneableAny> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Debug for dyn CloneableAny {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "CloneableAny({})", self.name())
    }
}

//pub type AnyParameters = Box<dyn Any + Send + Sync>;
pub type AnyParameters = Box<dyn CloneableAny>;

impl Clone for Box<dyn CloneableAny> {
    fn clone(&self) -> Self {
        (**self).clone_box()
    }
}

pub trait DowncastableAny {
    fn downcast_ref<T: 'static>(&self) -> Option<&T>;
    //fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T>;
}

impl DowncastableAny for Box<dyn CloneableAny> {
    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        (**self).as_any().downcast_ref::<T>()
    }
    //fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
    //    self.as_any_mut().downcast_mut::<T>()
    //}
}

pub trait AnyAugmentation: Send + Sync + Debug {
    fn name_any(&self) -> &str;
    fn sample_parameters_any(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> AnyParameters;

    fn maybe_sample_parameters_any(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> SampledParameters<AnyParameters>;

    fn augment_any(
        &self,
        waveform: &Array2<f32>,
        parameters: &AnyParameters,
    ) -> Result<Array2<f32>, String>;

    fn maybe_augment_any(
        &self,
        waveform: &Array2<f32>,
        parameters: SampledParameters<&AnyParameters>,
    ) -> Result<Array2<f32>, String>;

    fn clone_box(&self) -> Box<dyn AnyAugmentation>;
}

impl Clone for Box<dyn AnyAugmentation> {
    fn clone(&self) -> Self {
        self.as_ref().clone_box()
    }
}

impl<A: RandomAugmentation + Clone + 'static + Send + Sync + Debug> AnyAugmentation for A
where
    A::Parameters: 'static + Send + Sync + Clone,
{
    fn name_any(&self) -> &str {
        self.name()
    }
    fn sample_parameters_any(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> AnyParameters {
        let parameters = self.sample_parameters(audio, rng);
        Box::new(parameters)
    }

    fn maybe_sample_parameters_any(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> SampledParameters<AnyParameters> {
        let sampled_p: f32 = if let Some(rng) = rng.as_deref_mut() {
            rng.rand()
        } else {
            RandomNumberGenerator::new(None).rand()
        };
        if sampled_p < self.p() {
            SampledParameters::Sampled(self.sample_parameters_any(audio, rng))
        } else {
            SampledParameters::NotSampled
        }
    }

    fn augment_any(
        &self,
        waveform: &Array2<f32>,
        parameters: &AnyParameters,
    ) -> Result<Array2<f32>, String> {
        if let Some(parameters) = parameters.downcast_ref::<A::Parameters>() {
            return Ok(self.augment(waveform, parameters));
        } else {
            Err("Failed to downcast to A::Parameters".to_string())
        }
    }

    fn maybe_augment_any(
        &self,
        waveform: &Array2<f32>,
        parameters: SampledParameters<&AnyParameters>,
    ) -> Result<Array2<f32>, String> {
        match parameters {
            SampledParameters::NotSampled => Ok(waveform.to_owned()),
            SampledParameters::Sampled(parameters) => self.augment_any(waveform, parameters),
            SampledParameters::None => {
                let audio = Audio::new(waveform.clone(), 0);
                let sampled_parameters = self.sample_parameters_any(&audio, None);
                self.augment_any(&waveform, &sampled_parameters)
            }
        }
    }

    fn clone_box(&self) -> Box<dyn AnyAugmentation> {
        Box::new(self.clone())
    }
}
