from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from denoiser.data.augmentations.default import get_default_augmentation
from denoiser.data.stft import MelSpectrogram
from denoiser.data.collate import collate
from denoiser.data.dataset import AudioDataset
from denoiser.data.source import AudioSource
from denoiser.models.cfm.cfm import ConditionalFlowMatcher
from denoiser.models.codec.dac import DescriptAudioCodec
from denoiser.models.unet.unet import UNET1d, UNET1dDims


@dataclass
class TrainingConfig:
    speech_folder: str
    noise_folder: str
    codec_path: str

    sigma_0: float = 1.0
    sigma_1: float = 1e-7

    n_dim: int = 1024
    n_layers: int = 10

    sample_rate: int = 24_000  # mimi sample_rate
    sequence_length_n_tokens: int = 64
    batch_size: int = 64
    lr: float = 3e-4

    n_cfm_steps: int = 10
    max_steps: int = 1_000_000
    val_steps: int = 5_000
    smp_steps: int = 5_000
    n_val: int = 8192
    n_smp: int = 8

    n_workers: int = 8
    nocompile: bool = False
    noamp: bool = False


@dataclass
class Checkpoint:
    codec: str
    step: int
    best_loss: float
    model: dict[str, torch.Tensor]
    opt: dict[str, torch.Tensor]

    def __post_init__(self):
        self.executor = ThreadPoolExecutor(1)

    def save(self, path: str):
        def save():
            torch.save({k: getattr(self, k.name) for k in fields(self)}, path)

        self.executor.submit(save)

    @classmethod
    def load(cls, path: str, map_location: str | torch.device = "cpu"):
        checkpoint = torch.load(path, map_location=map_location)
        checkpoint = cls(**checkpoint)
        return cls


def train(exp_path: str, config: TrainingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    device_dtype = device.type

    # CODEC
    codec = DescriptAudioCodec(path=config.codec_path)
    codec = codec.eval()
    codec = codec.freeze()
    codec = codec.to(device)
    codec = torch.compile(codec, disable=not config.nocompile)
    mel_spectrogram = MelSpectrogram(1024, 256, 80, sample_rate=config.sample_rate)
    mel_spectrogram = mel_spectrogram.to(device)

    ### AUGMENTS
    sequence_length_s = config.sequence_length_n_tokens / codec.resolution_hz
    train_augments = get_default_augmentation(
        sequence_length_s=sequence_length_s, split="train", p=0.95
    )
    valid_augments = get_default_augmentation(
        sequence_length_s=sequence_length_s, split="valid", p=0.95
    )
    test_augments = get_default_augmentation(
        sequence_length_s=sequence_length_s, split="test", p=1.0
    )

    # SPEECH
    sr = config.sample_rate
    speech_folder = Path(config.speech_folder)
    train_audio_source = AudioSource(
        speech_folder / "index.train.json",
        sequence_length_s=sequence_length_s,
    )
    train_dataset = AudioDataset(
        train_audio_source,
        sample_rate=sr,
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
        sequence_length_s=sequence_length_s,
    )
    valid_dataset = AudioDataset(
        valid_audio_source,
        sample_rate=sr,
        augmentation=valid_augments,
    )
    valid_dataset = Subset(valid_dataset, list(range(config.n_val)))
    valid_dloader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        collate_fn=collate,
        num_workers=config.n_workers,
    )

    test_audio_source = AudioSource(
        speech_folder / "index.test.json",
        sequence_length_s=sequence_length_s,
    )
    test_dataset = AudioDataset(
        test_audio_source,
        sample_rate=sr,
        augmentation=test_augments,
    )
    test_dloader = DataLoader(
        test_dataset,
        batch_size=config.n_smp,
        collate_fn=collate,
    )

    unet_dims = UNET1dDims(
        in_dim=codec.dim,
        dim=config.n_dim,
        out_dim=codec.dim,
        n_layers=config.n_layers,
    )
    unet = UNET1d(unet_dims)
    unet = unet.to(device)
    unet = torch.compile(unet, disable=not config.nocompile)

    cflow_matcher = ConditionalFlowMatcher(
        unet,
        sigma_0=config.sigma_0,
        sigma_1=config.sigma_1,
    )
    cflow_matcher = cflow_matcher.to(device)

    opt = torch.optim.AdamW(cflow_matcher.parameters(), lr=config.lr)
    scaler = torch.GradScaler()

    def process_batch(batch):
        batch = batch.to(device)

        clean = batch.waveforms
        augmentation_params = batch.augmentation_params.to(device)
        noisy = train_augments.augment(clean, parameters=augmentation_params)
        with torch.autocast(device_type=device_dtype, enabled=config.noamp):
            with torch.no_grad():
                x_clean = 0.5 * codec.normalize(codec.encode(clean))
                x_noisy = 0.5 * codec.normalize(codec.encode(noisy))

        timestep = torch.rand((clean.shape[0],), device=device)
        with torch.autocast(device_type=device_dtype, enabled=config.noamp):
            vt, ut = cflow_matcher(
                x_1=x_clean,
                x_cond=x_noisy,
                timestep=timestep,
            )
            loss = torch.nn.functional.mse_loss(vt, ut)

        if cflow_matcher.training:
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad = torch.nn.utils.clip_grad_norm_(cflow_matcher.parameters(), 1e0)
            scaler.step(opt)
            scaler.update()

        metrics = {"loss": loss}
        if cflow_matcher.training:
            metrics["grad"] = grad
        return metrics

    @torch.inference_mode()
    def sample(batch):
        batch = batch.to(device)
        clean = batch.waveforms.clone()
        augmentation_params = batch.augmentation_params.to(device)
        noisy = test_augments.augment(clean, parameters=augmentation_params).clone()
        with torch.autocast(device_type=device_dtype, enabled=config.noamp):
            x_noisy = 0.5 * codec.normalize(codec.encode(noisy))

        x_0 = cflow_matcher.sigma_0 * torch.randn_like(x_noisy)
        timesteps = torch.linspace(0, 1, config.n_cfm_steps).tolist()
        x_cleaned = cflow_matcher.sample(
            x_0=x_0,
            x_cond=x_noisy,
            timesteps=timesteps,
        )
        with torch.no_grad():
            cleaned = codec.decode(2 * codec.unnormalize(x_cleaned))
        return noisy, cleaned

    @torch.inference_mode()
    def log_waveform(waveform, tag: str, step: int):
        writer.add_audio(f"{tag}.wav", waveform, step, sr)
        mels = mel_spectrogram(waveform[None].to(device)).log()
        mels = mels.flip(1)
        mels = (mels - mels.min()) / (mels.max() - mels.min())
        writer.add_image(f"{tag}.png", mels, step)

    smp_batch = next(iter(test_dloader)).to(device)
    writer = SummaryWriter(exp_path)
    pbar = tqdm(total=config.max_steps)
    for i, (clean, params) in enumerate(
        zip(smp_batch.waveforms, smp_batch.augmentation_params)
    ):
        with torch.no_grad():
            reconstructed = codec.reconstruct(clean[None])[0]
        noise = params.params[1].params[params.params[1].choice.item()].noise
        log_waveform(clean, f"2clean/{i}", 0)
        log_waveform(noise, f"3noise/{i}", 0)
        log_waveform(reconstructed, f"4reconstructed/{i}", 0)

    step, best_loss = 0, torch.inf
    while 1:
        for batch in train_dloader:

            if step % config.smp_steps == 0:
                cflow_matcher.eval()
                noisy, cleaned = sample(smp_batch)
                for i, (x, y_hat) in enumerate(zip(noisy, cleaned)):
                    log_waveform(y_hat, f"0cleaned/{i}", step)
                    if step == 0:
                        log_waveform(x, f"1noisy/{i}.wav", step)

            if step % config.val_steps == 0:
                vpbar = tqdm(total=len(valid_dloader), leave=False)
                cflow_matcher.eval()
                val_metrics = defaultdict(list)
                for vbatch in valid_dloader:
                    with torch.inference_mode():
                        metrics = process_batch(vbatch)
                    for key, value in metrics.items():
                        val_metrics[key] += [value]
                    vpbar.update(1)
                    vpbar.set_description_str(f"VALID {step} | {metrics['loss']:.4f}")
                del vbatch
                vpbar.close()

                val_metrics = {
                    k: torch.stack(v).mean().item() for k, v in val_metrics.items()
                }
                for key, values in val_metrics.items():
                    writer.add_scalar(f"valid/{key}", values, step)

                checkpoint = Checkpoint(
                    codec=codec.__class__.__name__,
                    step=step,
                    best_loss=best_loss,
                    model=cflow_matcher.state_dict(),
                    opt=opt.state_dict(),
                )
                checkpoint.save(Path(exp_path) / f"checkpoint.{step}.pt")
                if (loss := val_metrics["loss"]) < best_loss:
                    best_loss = loss
                    checkpoint.save(Path(exp_path) / "checkpoint.best.pt")

            cflow_matcher.train()
            metrics = process_batch(batch)

            pbar.set_description_str(f"TRAIN {step} | {metrics['loss']:.4f}")
            for key, val in metrics.items():
                writer.add_scalar(f"train/{key}", val, global_step=step)
            pbar.update(1)

            step += 1
            if step >= config.max_steps:
                checkpoint.executor.shutdown()
                pbar.close()
                return


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_arguments(TrainingConfig, dest="config")

    options = parser.parse_args()
    train(**vars(options))
