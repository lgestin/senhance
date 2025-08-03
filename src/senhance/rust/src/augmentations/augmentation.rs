use crate::audio::Audio;
use crate::augmentations::distributions::RandomNumberGenerator;
use ndarray::Array2;
use std::any::Any;
use std::fmt::{Debug, Formatter, Result as FmtResult};

pub trait Augments: Any {
    type Parameters;

    fn name(&self) -> &str;

    fn sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<Self::Parameters, String>;

    fn augment(&self, waveform: &Array2<f32>, parameters: &Self::Parameters) -> Array2<f32>;
}

#[derive(Debug)]
pub struct Augmentation<A: Augments> {
    augmentation: A,
}

impl<A: Augments> Augmentation<A> {
    pub fn new(augmentation: A) -> Self {
        Self { augmentation }
    }

    pub fn sample_parameters(
        &self,
        audio: &Audio,
        rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<A::Parameters, String> {
        match rng {
            Some(rng) => self.augmentation.sample_parameters(audio, Some(rng)),
            None => {
                let mut rng = RandomNumberGenerator::new(None);
                self.augmentation.sample_parameters(audio, Some(&mut rng))
            }
        }
    }

    pub fn augment(
        &self,
        waveform: &Array2<f32>,
        parameters: &Option<A::Parameters>,
    ) -> Result<Array2<f32>, String> {
        match parameters {
            Some(parameters) => Ok(self.augmentation.augment(waveform, parameters)),
            None => {
                let audio = Audio::new(waveform.clone(), None);
                let parameters = self.augmentation.sample_parameters(&audio, None)?;
                Ok(self.augmentation.augment(waveform, &parameters))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct RandomAugmentation<A: Augments> {
    pub augmentation: A,
    pub p: f32,
}

impl<A: Augments> RandomAugmentation<A> {
    pub fn new(augmentation: A, p: f32) -> Result<Self, String> {
        if p < 0.0 || p > 1.0 {
            Err("p must be between 0 and 1.".to_string())
        } else {
            Ok(Self { augmentation, p })
        }
    }

    pub fn sample_parameters(
        &self,
        audio: &Audio,
        mut rng: Option<&mut RandomNumberGenerator>,
    ) -> Result<Option<A::Parameters>, String> {
        if self.p == 1.0 {
            return Ok(Some(self.augmentation.sample_parameters(audio, rng)?));
        }
        let sampled_p: f32 = if let Some(rng) = rng.as_deref_mut() {
            rng.rand()
        } else {
            RandomNumberGenerator::new(None).rand()
        };
        if sampled_p < self.p {
            Ok(Some(self.augmentation.sample_parameters(audio, rng)?))
        } else {
            Ok(None)
        }
    }

    pub fn augment(
        &self,
        waveform: &Array2<f32>,
        parameters: &Option<A::Parameters>,
    ) -> Array2<f32> {
        match parameters {
            Some(parameters) => self.augmentation.augment(waveform, parameters),
            None => waveform.to_owned(),
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
    ) -> Result<AnyParameters, String>;

    fn augment_any(
        &self,
        waveform: &Array2<f32>,
        parameters: &AnyParameters,
    ) -> Result<Array2<f32>, String>;

    fn clone_box(&self) -> Box<dyn AnyAugmentation>;
}

impl Clone for Box<dyn AnyAugmentation> {
    fn clone(&self) -> Self {
        self.as_ref().clone_box()
    }
}

impl<A: Augments + Clone + 'static + Send + Sync + Debug> AnyAugmentation for A
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
    ) -> Result<AnyParameters, String> {
        let parameters = self.sample_parameters(audio, rng).unwrap();
        Ok(Box::new(parameters))
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

    fn clone_box(&self) -> Box<dyn AnyAugmentation> {
        Box::new(self.clone())
    }
}
