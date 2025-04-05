#!/usr/bin/env python

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from senhance.checkpoint_manager import CheckpointManager
from senhance.data.audio import Audio
from senhance.data.augmentations.augmentations import Augmentation
from senhance.data.collate import collate
from senhance.data.dataset import AudioDataset
from senhance.data.source import ArrowAudioSource
from senhance.data.stft import MelSpectrogram
from senhance.models.cfm.cfm import ConditionalFlowMatcher
from senhance.models.checkpoint import Checkpoint
from senhance.models.codec.dac import DescriptAudioCodec
from senhance.models.dit.dit import DiT, DiTDims
from senhance.models.unet.unet import UNET1d, UNET1dDims
from senhance.schedule_free_adamw import AdamWScheduleFree
from senhance.timestep_scheduler import LogNormTimestepScheduler
from senhance.trainer.logger import CompositeLogger, StdoutLogger, TensorBoardLogger
from senhance.trainer.sampler import Sampler
from senhance.trainer.trainer import (
    AMPDtype,
    TrainerConfig,
    TwoStreamsTrainer,
)


@dataclass
class TrainingConfig:
    speech_folder: str
    augmentation_config_path: str
    codec_path: str

    checkpoint_path: str = None

    model: str = "dit"
    n_dim: int = 1024
    n_layers: int = 10

    sequence_length_n_tokens: int = 64
    batch_size: int = 64
    lr: float = 3e-4
    t_lognorm_mean: float = -0.5
    t_lognorm_std: float = 1.0

    n_cfm_steps: int = 50
    max_steps: int = 1_000_000
    val_steps: int = 5_000
    smp_steps: int = 5_000
    checkpoint_steps: int = 25_000
    n_val: int = 8192
    n_smp: int = 8

    n_workers: int = 8
    nocompile: bool = False
    noamp: bool = False
    clip_grad_norm: float = 0.2

    def __post_init__(self):
        assert self.checkpoint_steps % self.val_steps == 0


def train(exp_path: str, config: TrainingConfig):
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    device_dtype = device.type

    # CODEC
    codec = DescriptAudioCodec(path=config.codec_path)
    codec = codec.eval()
    codec = codec.freeze()
    # Use float32 instead of bfloat16 to avoid NaN issues in codec
    codec = codec.to(device, dtype=torch.float32, non_blocking=True)
    # Disable compilation for codec to avoid NaN issues
    # codec.encode = torch.compile(
    #     codec.encode,
    #     disable=config.nocompile,
    #     # mode="max-autotune",
    #     # options={"triton.cudagraphs": False},
    # )
    sample_rate = codec.sample_rate
    Audio.stfter.to(device, non_blocking=True)

    # AUGMENTS
    sequence_length_s = config.sequence_length_n_tokens / codec.resolution_hz
    train_augments = Augmentation.from_yaml(
        path=config.augmentation_config_path,
        sequence_length_s=sequence_length_s,
        split="train",
    )
    valid_augments = Augmentation.from_yaml(
        path=config.augmentation_config_path,
        sequence_length_s=sequence_length_s,
        split="valid",
    )
    smp_augments = Augmentation.from_yaml(
        path=config.augmentation_config_path,
        sequence_length_s=sequence_length_s,
        split="test",
    )

    # SPEECH
    speech_folder = Path(config.speech_folder)
    train_audio_source = ArrowAudioSource(
        speech_folder / "data.train.24000hz.arrow",
        sequence_length_s=sequence_length_s,
        is_speech=True,
    )
    train_dataset = AudioDataset(
        train_audio_source,
        sample_rate=sample_rate,
        augmentation=train_augments,
    )
    train_dloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=collate,
        num_workers=config.n_workers,
        shuffle=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    valid_audio_source = ArrowAudioSource(
        speech_folder / "data.valid.24000hz.arrow",
        sequence_length_s=sequence_length_s,
        is_speech=True,
    )
    valid_dataset = AudioDataset(
        valid_audio_source,
        sample_rate=sample_rate,
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
        Path("/data/denoising/speech/daps/clean") / "data.test.24000hz.arrow",
        sequence_length_s=sequence_length_s,
        is_speech=True,
    )
    test_dataset = AudioDataset(
        test_audio_source,
        sample_rate=sample_rate,
        augmentation=smp_augments,
    )
    smp_dloader = DataLoader(
        test_dataset,
        batch_size=config.n_smp,
        collate_fn=collate,
    )

    # MODEL
    if config.model == "unet":
        unet_dims = UNET1dDims(
            in_dim=codec.dim,
            dim=config.n_dim,
            t_dim=config.n_dim,
        )
        unet = UNET1d(unet_dims)
        unet = unet.to(device, non_blocking=True)
        denoiser = unet

    elif config.model == "dit":
        dit_dims = DiTDims(
            in_dim=codec.dim,
            dim=config.n_dim,
            t_dim=config.n_dim,
            n_layers=config.n_layers,
        )
        dit = DiT(dit_dims)
        dit = dit.to(device, non_blocking=True)
        denoiser = dit

    # FLOW MATCHER
    cflow_matcher = ConditionalFlowMatcher(denoiser)
    cflow_matcher = cflow_matcher.to(device, non_blocking=True)

    # OPTIMIZER
    opt = AdamWScheduleFree(
        cflow_matcher.parameters(),
        lr=config.lr,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        warmup_steps=250,
    )

    # SCALER
    scaler = torch.GradScaler() if not config.noamp else None

    # LOAD CHECKPOINT
    step, best_loss = 0, torch.inf
    if config.checkpoint_path:
        checkpoint = Checkpoint.load(config.checkpoint_path, map_location=device)
        step, best_loss = checkpoint.step, checkpoint.best_loss
        denoiser.load_state_dict(checkpoint.model)
        opt.load_state_dict(checkpoint.opt)
        if scaler is not None:
            scaler.load_state_dict(checkpoint.scaler)

    # COMPILE
    denoiser = torch.compile(denoiser, disable=config.nocompile)

    # TIMESTEP SCHEDULER
    t_scheduler = LogNormTimestepScheduler(
        mean=config.t_lognorm_mean, std=config.t_lognorm_std
    )

    # LOGGER
    mel_spectrogram = MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        n_mels=80,
        sample_rate=sample_rate,
    )
    mel_spectrogram = mel_spectrogram.to(device, non_blocking=True)
    tb_logger = TensorBoardLogger(
        log_dir=exp_path,
        mel_spectrogram=mel_spectrogram,
        sample_rate=sample_rate,
    )
    rich_logger = StdoutLogger(
        total_steps=config.max_steps,
        initial_step=step,
        unit="batch",
        # refresh_per_second=4,
        # config=config,
    )
    logger = CompositeLogger(
        loggers=[tb_logger, rich_logger],
        mel_spectrogram=mel_spectrogram,
    )

    # CUDA STREAMS
    codec_stream = torch.cuda.Stream(device=device)
    denoiser_stream = torch.cuda.Stream(device=device)

    # TRAINER CONFIG
    trainer_config = TrainerConfig(
        clip_grad_norm=config.clip_grad_norm,
        device=device,
        amp_dtype=AMPDtype.BF16 if device_dtype == "cuda" else AMPDtype.FP32,
        max_steps=config.max_steps,
        checkpoint_steps=config.checkpoint_steps,
        smp_steps=config.smp_steps,
        valid_steps=config.val_steps,
        noamp=config.noamp,
        n_cfm_steps=config.n_cfm_steps,
        n_smp=config.n_smp,
    )

    # CREATE TRAINER
    checkpoint_manager = CheckpointManager(exp_path=exp_path)
    trainer = TwoStreamsTrainer(
        config=trainer_config,
        codec=codec,
        model=cflow_matcher,
        t_scheduler=t_scheduler,
        optimizer=opt,
        scaler=scaler,
        logger=logger,
        train_dloader=train_dloader,
        valid_dloader=valid_dloader,
        smp_dloader=smp_dloader,
        train_augments=train_augments,
        valid_augments=valid_augments,
        smp_augments=smp_augments,
        checkpoint_manager=checkpoint_manager,
        sampler=Sampler(),
        state=None,
    )

    print(f"Starting training with model: {denoiser}")
    print(f"Total parameters: {sum(p.numel() for p in denoiser.parameters()):,}")

    # RUN TRAINING
    trainer.train()


if __name__ == "__main__":
    import simple_parsing

    parser = simple_parsing.ArgumentParser()
    parser.add_argument("--exp_path", type=str, required=True)
    parser.add_arguments(TrainingConfig, dest="config")

    options = parser.parse_args()
    train(**vars(options))
