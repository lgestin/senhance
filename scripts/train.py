import torch

from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from denoiser.data.source import AudioSource
from denoiser.data.dataset import AudioDataset
from denoiser.data.collate import collate
from denoiser.data.augmentations import BackgroundNoise

from denoiser.models.codec.mimi import MimiCodec
from denoiser.cfm import ConditionalFlowMatcher
from denoiser.models.unet.unet import UNET1d


@dataclass
class TrainingConfig:
    speech_folder: str
    noise_folder: str
    codec_path: str

    sigma_0: float = 1.0
    sigma_1: float = 1e-7

    sample_rate: int = 24_000  # mimi sample_rate
    sequence_length_s: int = 64 / 12.5
    batch_size: int = 64
    lr: float = 1e-3

    n_cfm_steps: int = 10
    max_steps: int = 1_000_000
    smp_steps: int = 5_000
    n_smp: int = 8

    n_workers: int = 8
    nocompile: bool = False
    noamp: bool = False


def train(exp_path: str, config: TrainingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    device_dtype = device.type

    ### AUGMENTS
    noise_folder = Path(config.noise_folder)
    train_augments = BackgroundNoise(
        noise_index_path=noise_folder / "index.train.json",
        min_snr=-5,
        max_snr=25,
        min_duration_s=config.sequence_length_s,
    )
    valid_augments = BackgroundNoise(
        noise_index_path=noise_folder / "index.valid.json",
        min_snr=-5,
        max_snr=25,
        min_duration_s=config.sequence_length_s,
    )
    test_augments = BackgroundNoise(
        noise_index_path=noise_folder / "index.test.json",
        min_snr=-5,
        max_snr=25,
        min_duration_s=config.sequence_length_s,
    )

    # SPEECH
    speech_folder = Path(config.speech_folder)
    train_audio_source = AudioSource(
        speech_folder / "index.train.json",
        sequence_length_s=config.sequence_length_s,
    )
    train_dataset = AudioDataset(
        train_audio_source,
        sample_rate=config.sample_rate,
        augmentation=train_augments,
    )
    train_dloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate,
        num_workers=config.n_workers,
        shuffle=True,
    )

    valid_audio_source = AudioSource(
        speech_folder / "index.valid.json",
        sequence_length_s=config.sequence_length_s,
    )
    valid_dataset = AudioDataset(
        train_audio_source,
        sample_rate=config.sample_rate,
        augmentation=valid_augments,
    )
    valid_dloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        collate_fn=collate,
    )

    test_audio_source = AudioSource(
        speech_folder / "index.test.json",
        sequence_length_s=config.sequence_length_s,
    )
    test_dataset = AudioDataset(
        test_audio_source,
        sample_rate=config.sample_rate,
        augmentation=test_augments,
    )
    test_dloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collate,
    )

    codec = MimiCodec(safetensors_path=config.codec_path)
    codec = codec.eval()
    codec = codec.freeze()
    codec = codec.to(device)
    codec = torch.compile(codec, disable=not config.nocompile)

    unet = UNET1d(
        in_channels=codec.dim,
        hidden_channels=codec.dim,
        out_channels=codec.dim,
    )
    unet = unet.to(device)
    unet = torch.compile(unet, disable=not config.nocompile)

    cflow_matcher = ConditionalFlowMatcher(
        unet,
        sigma_0=config.sigma_0,
        sigma_1=config.sigma_1,
    )
    cflow_matcher = cflow_matcher.to(device)

    opt = torch.optim.AdamW(unet.parameters(), lr=config.lr)
    scaler = torch.GradScaler()

    def process_batch(batch):
        batch = batch.to(device)

        clean = batch.waveforms
        noisy = train_augments.augment(clean, parameters=batch.augmentation_params)
        with torch.no_grad():
            with torch.autocast(device_type=device_dtype, enabled=config.noamp):
                x_clean = codec.encode(clean)
                x_noisy = codec.encode(noisy)

        t = torch.rand(clean.shape[0], 1, 1, device=device)
        with torch.autocast(device_type=device_dtype, enabled=config.noamp):
            vt, ut = cflow_matcher(t, x_1=x_clean, x_cond=x_noisy)
            loss = torch.nn.functional.mse_loss(vt, ut)

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        return {"loss": loss.item()}

    @torch.inference_mode()
    def sample(batch):
        batch = batch.to(device)
        clean = batch.waveforms.clone()
        noisy = test_augments.augment(
            clean, parameters=batch.augmentation_params
        ).clone()
        with torch.no_grad():
            x_noisy = codec.encode(noisy)

        x_0 = cflow_matcher.sigma_0 * torch.randn_like(x_noisy)
        x_cleaned = cflow_matcher.sample(
            x_0=x_0,
            x_cond=x_noisy,
            n_steps=config.n_cfm_steps,
        )[0]
        with torch.no_grad():
            cleaned = codec.decode(x_cleaned)
        return cleaned, clean, noisy

    writer = SummaryWriter(exp_path)
    pbar = tqdm(total=config.max_steps)
    smp_batch = next(iter(test_dloader))

    step = 0
    while 1:
        for batch in train_dloader:

            if step % config.smp_steps == 0:
                clean, cleaned, noisy = sample(smp_batch)
                for i, (y_hat, y, x) in enumerate(zip(clean, cleaned, noisy)):
                    writer.add_audio(f"smp/{i}.clean.wav", y_hat, step)
                    writer.add_audio(f"smp/{i}.cleaned.wav", y, step)
                    writer.add_audio(f"smp/{i}.noisy.wav", x, step)

            cflow_matcher.train()
            metrics = process_batch(batch)

            pbar.set_description_str(f"TRAIN {step} | {metrics['loss']:.4f}")
            for key, val in metrics.items():
                writer.add_scalar(f"train/{key}", val, global_step=step)
            pbar.update(1)

            step += 1
            if step >= config.max_steps:
                pbar.close()
                return


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_arguments(TrainingConfig, dest="config")

    options = parser.parse_args()
    train(**vars(options))
