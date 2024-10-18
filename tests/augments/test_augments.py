from concurrent.futures import ThreadPoolExecutor
from denoiser.data.augmentations.default import get_default_augmentation
from denoiser.data.audio import Audio
from denoiser.data.augmentations.augmentations import BatchAugmentationParameters


def test_augments():
    audio = Audio("/data/denoising/speech/daps/clean/f10_script1_clean.wav")
    augment = get_default_augmentation(1, split="train", p=1.0)

    with ThreadPoolExecutor(4) as executor:
        futures = [executor.submit(audio.salient_excerpt, 1) for _ in range(16)]
        excerpts = [future.result() for future in futures]

    augment_params = [augment.sample_parameters(excerpt) for excerpt in excerpts]
    # augment_params = augment_params[0].collate(augment_params)
    augment_params = BatchAugmentationParameters.collate(augment_params)


if __name__ == "__main__":
    test_augments()
