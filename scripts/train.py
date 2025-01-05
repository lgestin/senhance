import re
import shlex
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from senhance.data.augmentations.default import get_default_augmentation
from senhance.data.collate import collate
from senhance.data.dataset import AudioDataset
from senhance.data.source import ArrowAudioSource
from senhance.data.stft import MelSpectrogram
from senhance.models.cfm.cfm import ConditionalFlowMatcher
from senhance.models.checkpoint import Checkpoint
from senhance.models.codec.dac import DescriptAudioCodec
from senhance.models.unet.simple_unet import UNET1d, UNET1dDims


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


def train(exp_path: str, config: TrainingConfig):
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    device_dtype = device.type

    # CODEC
    codec = DescriptAudioCodec(path=config.codec_path)
    codec = codec.eval()
    codec = codec.freeze()
    codec = codec.to(device)
    codec = torch.compile(codec, disable=not config.nocompile)
    mel_spectrogram = MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        sample_rate=config.sample_rate,
    )
    mel_spectrogram = mel_spectrogram.to(device)

    ### AUGMENTS
    sequence_length_s = config.sequence_length_n_tokens / codec.resolution_hz
    train_augments = get_default_augmentation(
        noise_folder="/data/denoising/noise/",
        sample_rate=config.sample_rate,
        sequence_length_s=sequence_length_s,
        split="train",
        p=0.95,
    )
    valid_augments = get_default_augmentation(
        noise_folder="/data/denoising/noise/",
        sample_rate=config.sample_rate,
        sequence_length_s=sequence_length_s,
        split="valid",
        p=0.95,
    )
    test_augments = get_default_augmentation(
        noise_folder="/data/denoising/noise/",
        sample_rate=config.sample_rate,
        sequence_length_s=sequence_length_s,
        split="test",
        p=1.0,
    )

    # SPEECH
    sr = config.sample_rate
    speech_folder = Path(config.speech_folder)
    train_audio_source = ArrowAudioSource(
        speech_folder / "data.train.arrow",
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

    valid_audio_source = ArrowAudioSource(
        speech_folder / "data.valid.arrow",
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

    test_audio_source = ArrowAudioSource(
        speech_folder / "data.test.arrow",
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
        t_dim=config.n_dim,
    )
    unet = UNET1d(unet_dims)
    unet = unet.to(device)
    unet = torch.compile(unet, disable=not config.nocompile)

    cflow_matcher = ConditionalFlowMatcher(unet)
    cflow_matcher = cflow_matcher.to(device)

    opt = torch.optim.AdamW(
        cflow_matcher.parameters(),
        lr=config.lr,
        betas=(0.9, 0.99),
    )
    scaler = torch.GradScaler()

    def process_batch(batch):
        batch = batch.to(device)

        clean = batch.waveforms
        augmentation_params = batch.augmentation_params
        if augmentation_params:
            augmentation_params = augmentation_params.to(device)
        noisy = clean.clone()
        noisy = train_augments.augment(noisy, parameters=augmentation_params)
        with torch.autocast(device_type=device_dtype, enabled=config.noamp):
            with torch.no_grad():
                x_clean, x_noisy = codec.normalize(
                    codec.encode(torch.cat([clean, noisy]))
                ).chunk(2, dim=0)

        timestep = torch.rand((clean.shape[0],), device=device)
        with torch.autocast(device_type=device_dtype, enabled=config.noamp):
            vt, ut = cflow_matcher(
                x_0=x_noisy,
                x_1=x_clean,
                timestep=timestep,
            )
            loss = torch.nn.functional.mse_loss(vt, ut)

        if cflow_matcher.training:
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad = torch.nn.utils.clip_grad_norm_(
                cflow_matcher.parameters(),
                max_norm=1e0,
            )
            scaler.step(opt)
            scaler.update()

        metrics = {"loss": loss}
        if cflow_matcher.training:
            metrics["grad"] = grad
        return metrics

    @torch.inference_mode()
    def sample(batch):
        batch = batch.to(device)
        clean = batch.waveforms
        noisy = clean.clone()
        augmentation_params = batch.augmentation_params
        if augmentation_params:
            augmentation_params = augmentation_params.to(device)
        noisy = test_augments.augment(
            waveform=clean.clone(),
            parameters=augmentation_params,
        )
        with torch.autocast(device_type=device_dtype, enabled=config.noamp):
            x_noisy = codec.normalize(codec.encode(noisy))

        timesteps = torch.linspace(0, 1, config.n_cfm_steps).tolist()
        x_cleaned = cflow_matcher.sample(x_0=x_noisy, timesteps=timesteps)
        with torch.no_grad():
            cleaned = codec.decode(codec.unnormalize(x_cleaned))
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
    for i, (clean, params) in enumerate(
        zip(smp_batch.waveforms, smp_batch.augmentation_params)
    ):
        with torch.no_grad():
            reconstructed = codec.reconstruct(clean[None])[0]
        log_waveform(clean, f"{i}/clean", 0)
        log_waveform(reconstructed, f"{i}/reconstructed", 0)

    pbar = tqdm(total=config.max_steps, unit="batch", smoothing=0.1)
    step, best_loss = 0, torch.inf
    while step < config.max_steps:
        for batch in train_dloader:
            if step % config.smp_steps == 0:
                cflow_matcher.eval()
                noisy, cleaned = sample(smp_batch)
                for i, (x, y_hat, clean) in enumerate(
                    zip(noisy, cleaned, smp_batch.waveforms)
                ):
                    log_waveform(y_hat, f"{i}/cleaned", step)
                    if step == 0:
                        log_waveform(x - clean, f"{i}/noise", step)
                        log_waveform(x, f"{i}/noisy", step)

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
                    vpbar.set_description_str(
                        f"VALID {step} | {metrics['loss']:.4f}"
                    )
                del vbatch
                vpbar.close()

                val_metrics = {
                    k: torch.stack(v).mean().item()
                    for k, v in val_metrics.items()
                }
                for key, values in val_metrics.items():
                    writer.add_scalar(f"valid/{key}", values, step)

                checkpoint = Checkpoint(
                    codec=codec.__class__.__name__,
                    step=step,
                    best_loss=best_loss,
                    dims=cflow_matcher.module.dims,
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
            shm_size = shlex.split("df -h /dev/shm")
            shm_size = subprocess.run(
                shm_size, stdout=subprocess.PIPE
            ).stdout.decode("utf-8")
            shm_size = float(re.findall(r"\s(\d)%\s", shm_size)[0])
            writer.add_scalar("train/shm_size", shm_size, global_step=step)
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
