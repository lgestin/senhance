from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from senhance.data.augmentations.augmentations import Augmentation
from senhance.models.cfm.cfm import ConditionalFlowMatcher
from senhance.models.codec.codec import Codec


@dataclass
class Sampled:
    sample_rate: int
    clean: torch.Tensor
    noisy: torch.Tensor
    cleaned: torch.Tensor
    cleaned_sequence: torch.Tensor

    @property
    def batch_size(self):
        return self.clean.size(0)


class Sampler:
    @torch.inference_mode()
    def sample(
        self,
        model: ConditionalFlowMatcher,
        codec: Codec,
        n_samples: int,
        smp_dloader: DataLoader,
        smp_augments: Augmentation,
        n_cfm_steps: int,
        noamp: bool = False,
    ) -> Sampled:
        """Perform sampling and log results (single stream)."""
        model.eval()

        # Get a sample batch
        smp_batch = next(iter(smp_dloader)).to(model.device)

        clean = smp_batch.waveforms
        augmentation_params = smp_batch.augmentation_params
        if augmentation_params is not None:
            augmentation_params = augmentation_params.to(
                model.device, non_blocking=True
            )

        # Apply augmentation
        noisy = smp_augments.augment(
            waveform=clean.clone(),
            parameters=augmentation_params,
        )

        # Process samples one at a time to avoid OOM
        n_samples = min(len(clean), n_samples)
        timesteps = torch.linspace(0, 1, n_cfm_steps, device=model.device)

        cleaned = []
        for i in range(n_samples):
            with torch.autocast(
                device_type=model.device.type,
                dtype=torch.float16,
                enabled=not noamp,
            ):
                # Encode noisy audio for single sample
                x_noisy_i = codec.normalize(codec.encode(noisy[i : i + 1]))

                # Sample with flow matcher
                x_cleaned_i = model.sample(
                    x_0=x_noisy_i,
                    timesteps=timesteps,
                    return_intermediates=True,
                )

                # Decode all timesteps for this sample
                x_cleaned_flat = x_cleaned_i.squeeze(1)
                cleaned_flat = codec.decode(codec.unnormalize(x_cleaned_flat))
                cleaned_sequence = cleaned_flat  # [n_timesteps, channels, samples]
                cleaned.append(cleaned_sequence)

        cleaned = torch.stack(cleaned)
        sampled = Sampled(
            sample_rate=smp_batch.audios[0].sample_rate,
            clean=clean,
            noisy=noisy,
            cleaned=cleaned[:, -1],
            cleaned_sequence=cleaned,
        )
        return sampled
