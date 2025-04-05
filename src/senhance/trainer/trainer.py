from dataclasses import dataclass
from enum import Enum

import torch
from torch import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from senhance.checkpoint_manager import CheckpointManager
from senhance.data.augmentations.augmentations import Augmentation
from senhance.data.dataset import Batch, FlowMatchingBatch
from senhance.models.cfm.cfm import ConditionalFlowMatcher
from senhance.models.codec.codec import Codec
from senhance.timestep_scheduler import TimestepScheduler
from senhance.trainer.logger import Logger, ValidationLogger
from senhance.trainer.sampler import Sampler


@dataclass
class TrainerState:
    step: int
    best_loss: float = float("inf")


class AMPDtype(str, Enum):
    FP16 = "fp16"
    BF16 = "bf16"
    FP32 = "fp32"

    @property
    def dtype(self):
        if self == AMPDtype.FP16:
            return torch.float16
        elif self == AMPDtype.BF16:
            return torch.bfloat16
        elif self == AMPDtype.FP32:
            return torch.float32


@dataclass
class TrainerConfig:
    clip_grad_norm: float | None
    device: torch.device
    amp_dtype: AMPDtype
    smp_steps: int
    valid_steps: int
    checkpoint_steps: int
    max_steps: int
    noamp: bool
    n_cfm_steps: int = 50  # For sampling
    n_smp: int = 8  # Number of samples to process and log


class Trainer:
    def __init__(self, config: TrainerConfig, state: TrainerState | None = None):
        self.config = config
        self.state = state or TrainerState(step=0)

    def prepare_batch(self, batch: Batch) -> FlowMatchingBatch:
        raise NotImplementedError

    def process_batch(self, batch: FlowMatchingBatch) -> dict:
        raise NotImplementedError

    @property
    def device(self):
        return self.config.device

    @property
    def amp_dtype(self):
        return self.config.amp_dtype.dtype

    @property
    def smp_steps(self):
        return self.config.smp_steps

    @property
    def valid_steps(self):
        return self.config.valid_steps

    @property
    def checkpoint_steps(self):
        return self.config.checkpoint_steps

    @property
    def max_steps(self):
        return self.config.max_steps

    @property
    def noamp(self):
        return self.config.noamp

    @property
    def step(self):
        return self.state.step

    @property
    def best_loss(self):
        return self.state.best_loss


class SingleStreamTrainer(Trainer):
    def __init__(
        self,
        config: TrainerConfig,
        codec: Codec,
        model: ConditionalFlowMatcher,
        t_scheduler: TimestepScheduler,
        optimizer: Optimizer,
        scaler: GradScaler | None,
        logger: Logger,
        train_dloader: DataLoader,
        valid_dloader: DataLoader,
        smp_dloader: DataLoader,
        train_augments: Augmentation,
        valid_augments: Augmentation,
        smp_augments: Augmentation,
        state: TrainerState | None,
        checkpoint_manager: CheckpointManager,
        sampler: Sampler,
    ):
        super().__init__(config=config, state=state)
        self.codec = codec
        self.model = model
        self.t_scheduler = t_scheduler
        self.optimizer = optimizer
        self.scaler = scaler
        self.logger = logger

        self.train_dloader = train_dloader
        self.valid_dloader = valid_dloader
        self.smp_dloader = smp_dloader

        self.train_augments = train_augments
        self.valid_augments = valid_augments
        self.smp_augments = smp_augments

        self.checkpoint_manager = checkpoint_manager
        self.sampler = sampler

    def train(self):
        """Main training loop – single CUDA stream, no overlap."""
        # Log initial samples once
        self._log_initial_samples()

        while self.step < self.max_steps:
            for batch in self.train_dloader:
                # Sampling step
                if self.step % self.smp_steps == 0:
                    self.optimizer.eval()
                    sampled = self.sampler.sample(
                        model=self.model,
                        codec=self.codec,
                        smp_dloader=self.smp_dloader,
                        smp_augments=self.smp_augments,
                        n_cfm_steps=self.config.n_cfm_steps,
                        n_samples=self.config.n_cfm_steps,
                    )
                    self.logger.log_sampled(sampled, step=self.step)

                # Validation step
                if self.step % self.valid_steps == 0:
                    self.validation(valid_dloader=self.valid_dloader, step=self.step)

                # Checkpoint step
                if self.step % self.checkpoint_steps == 0 and self.step > 0:
                    self.checkpoint_manager.save(
                        step=self.step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        best_loss=self.best_loss,
                    )

                # Regular train step
                codec_features = self.prepare_batch(batch)

                if hasattr(self.optimizer, "train"):
                    self.optimizer.train()
                self.model.train()

                metrics = self.process_batch(codec_features)
                self.logger.log_metrics(metrics, self.step, prefix="train")

                self.logger.update_progress(1)
                self.state.step += 1

                if self.step >= self.max_steps:
                    self.logger.close()
                    return

    # ------------ core steps shared by all trainers ------------

    @torch.no_grad()
    def prepare_batch(self, batch: Batch) -> FlowMatchingBatch:
        """Prepare batch for processing (codec encoding). Single-stream version."""

        # assume batch.waveforms is CPU + (optionally) pinned by DataLoader
        clean = batch.waveforms.to(self.device, non_blocking=True)
        augmentation_params = batch.augmentation_params
        if augmentation_params is not None:
            augmentation_params = augmentation_params.to(self.device, non_blocking=True)

        noisy = clean.clone()
        noisy = self.train_augments.augment(noisy, parameters=augmentation_params)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            encoded = self.codec.encode(torch.cat([clean, noisy], dim=0))
            normalized = self.codec.normalize(encoded)
            x_1, x_0 = normalized.chunk(2, dim=0)

        timestep = self.t_scheduler.sample(
            batch_size=clean.size(0),
            device=self.device,
        )

        return FlowMatchingBatch(timestep=timestep, x_0=x_0, x_1=x_1)

    def process_batch(self, batch: FlowMatchingBatch) -> dict[str, torch.Tensor]:
        """Process a batch and compute metrics."""
        metrics: dict[str, torch.Tensor] = {}

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            u_t, path_sample = self.model(
                x_0=batch.x_0, x_1=batch.x_1, timestep=batch.timestep
            )
            loss = torch.nn.functional.mse_loss(path_sample.dx_t, u_t)

        metrics["loss"] = loss

        if self.model.training:
            self.optimizer.zero_grad()

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            else:
                loss.backward()

            if self.config.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.clip_grad_norm,
                )
                metrics["grad_norm"] = grad_norm

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        return metrics

    @torch.inference_mode()
    def validation(self, valid_dloader: DataLoader, step: int):
        """Perform validation and log metrics (single stream)."""
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        val_logger = ValidationLogger(total=len(self.valid_dloader))

        for vbatch in valid_dloader:
            codec_features = self.prepare_batch(vbatch)
            metrics = self.process_batch(codec_features)
            val_logger.update(metrics, step)

        val_metrics = val_logger.get_averaged_metrics()
        val_logger.close()

        # Log validation metrics
        self.logger.log_metrics(val_metrics, step, prefix="valid")

        # Update best loss
        loss_val = val_metrics.get("loss", float("inf"))
        if loss_val < self.best_loss:
            self.state.best_loss = loss_val

    def _log_initial_samples(self):
        """Log initial clean and reconstructed samples."""
        smp_batch = next(iter(self.smp_dloader)).to(self.device)

        n_samples = min(len(smp_batch.waveforms), self.config.n_smp)
        for i, waveform in enumerate(smp_batch.waveforms[:n_samples]):
            # Log clean
            self.logger.log_audio(
                f"{i}/clean",
                waveform,
                0,
                self.codec.sample_rate,
            )

            # Log reconstructed
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=not self.noamp,
            ):
                reconstructed = self.codec.reconstruct(waveform[None])[0]

            self.logger.log_audio(
                f"{i}/reconstructed",
                reconstructed,
                0,
                self.codec.sample_rate,
            )

    def _log_initial_samples(self):
        """Log initial clean and reconstructed samples."""
        smp_batch = next(iter(self.smp_dloader)).to(self.device)

        n_samples = min(len(smp_batch.waveforms), self.config.n_smp)
        for i, waveform in enumerate(smp_batch.waveforms[:n_samples]):
            # Log clean
            self.logger.log_audio(
                f"{i}/clean",
                waveform,
                0,
                self.codec.sample_rate,
            )

            # Log reconstructed
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=not self.noamp,
            ):
                reconstructed = self.codec.reconstruct(waveform[None])[0]

            self.logger.log_audio(
                f"{i}/reconstructed",
                reconstructed,
                0,
                self.codec.sample_rate,
            )


# ----------------------------------------------------------------------
# Profiling trainers: stripped down for performance analysis with nsys
# ----------------------------------------------------------------------
class ProfilingTrainer(Trainer):
    """Minimal trainer for profiling with nsys.

    Strips away validation, sampling, checkpointing, and logging to focus
    on core training loop performance. Useful for identifying bottlenecks.
    """

    def __init__(
        self,
        config: TrainerConfig,
        codec: Codec,
        model: ConditionalFlowMatcher,
        t_scheduler: TimestepScheduler,
        optimizer: Optimizer,
        scaler: GradScaler | None,
        train_dloader: DataLoader,
        train_augments: Augmentation,
        state: TrainerState | None = None,
    ):
        super().__init__(config=config, state=state)
        self.codec = codec
        self.model = model
        self.t_scheduler = t_scheduler
        self.optimizer = optimizer
        self.scaler = scaler
        self.train_dloader = train_dloader
        self.train_augments = train_augments

    def train(self):
        """Minimal training loop – single CUDA stream."""
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        self.model.train()

        while self.step < self.max_steps:
            for batch in self.train_dloader:
                codec_features = self.prepare_batch(batch)
                self.process_batch(codec_features)

                self.state.step += 1

                if self.step >= self.max_steps:
                    return

    @torch.no_grad()
    def prepare_batch(self, batch: Batch) -> FlowMatchingBatch:
        """Prepare batch for processing (codec encoding)."""
        torch.cuda.nvtx.range_push("prepare_batch")

        torch.cuda.nvtx.range_push("data_transfer")
        clean = batch.waveforms.to(self.device, non_blocking=True)
        augmentation_params = batch.augmentation_params
        if augmentation_params is not None:
            augmentation_params = augmentation_params.to(self.device, non_blocking=True)
        torch.cuda.nvtx.range_pop()  # data_transfer

        torch.cuda.nvtx.range_push("augmentation")
        noisy = clean.clone()
        noisy = self.train_augments.augment(noisy, parameters=augmentation_params)
        torch.cuda.nvtx.range_pop()  # augmentation

        torch.cuda.nvtx.range_push("codec_encode")
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            encoded = self.codec.encode(torch.cat([clean, noisy], dim=0))
            normalized = self.codec.normalize(encoded)
            x_1, x_0 = normalized.chunk(2, dim=0)
        torch.cuda.nvtx.range_pop()  # codec_encode

        torch.cuda.nvtx.range_push("sample_timestep")
        timestep = self.t_scheduler.sample(
            batch_size=clean.size(0),
            device=self.device,
        )
        torch.cuda.nvtx.range_pop()  # sample_timestep

        torch.cuda.nvtx.range_pop()  # prepare_batch
        return FlowMatchingBatch(timestep=timestep, x_0=x_0, x_1=x_1)

    def process_batch(self, batch: FlowMatchingBatch) -> dict[str, torch.Tensor]:
        """Process a batch and compute metrics."""
        torch.cuda.nvtx.range_push("train_step")

        torch.cuda.nvtx.range_push("forward")
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=not self.noamp,
        ):
            u_t, path_sample = self.model(
                x_0=batch.x_0, x_1=batch.x_1, timestep=batch.timestep
            )
            loss = torch.nn.functional.mse_loss(path_sample.dx_t, u_t)
        torch.cuda.nvtx.range_pop()  # forward

        torch.cuda.nvtx.range_push("backward")
        self.optimizer.zero_grad()

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        torch.cuda.nvtx.range_pop()  # backward

        torch.cuda.nvtx.range_push("optimizer")
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)

        if self.config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.clip_grad_norm,
            )

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        torch.cuda.nvtx.range_pop()  # optimizer

        torch.cuda.nvtx.range_pop()  # train_step
        return {"loss": loss}


# ----------------------------------------------------------------------
# Two-stream trainer: inherits from single-stream and overrides train
# ----------------------------------------------------------------------
class TwoStreamsTrainer(SingleStreamTrainer):
    def __init__(
        self,
        config: TrainerConfig,
        codec: Codec,
        model,
        t_scheduler: TimestepScheduler,
        optimizer,
        scaler,
        logger: Logger,
        train_dloader,
        valid_dloader,
        smp_dloader,
        train_augments,
        valid_augments,
        smp_augments,
        state: TrainerState | None = None,
        checkpoint_manager=None,
        sampler=None,
    ):
        super().__init__(
            config=config,
            codec=codec,
            model=model,
            t_scheduler=t_scheduler,
            optimizer=optimizer,
            scaler=scaler,
            logger=logger,
            train_dloader=train_dloader,
            valid_dloader=valid_dloader,
            smp_dloader=smp_dloader,
            train_augments=train_augments,
            valid_augments=valid_augments,
            smp_augments=smp_augments,
            state=state,
            checkpoint_manager=checkpoint_manager,
            sampler=sampler,
        )

        self.codec_stream = torch.cuda.Stream(device=model.device)
        self.denoiser_stream = torch.cuda.current_stream(device=model.device)

    def train(self):
        """Main training loop – two CUDA streams with overlap."""
        codec_features = None

        # Log initial samples
        self._log_initial_samples()

        while self.step < self.max_steps:
            for batch in self.train_dloader:
                # Sampling step
                if self.step % self.smp_steps == 0:
                    self.optimizer.eval()
                    sampled = self.sampler.sample(
                        model=self.model,
                        codec=self.codec,
                        smp_dloader=self.smp_dloader,
                        smp_augments=self.smp_augments,
                        n_cfm_steps=self.config.n_cfm_steps,
                        n_samples=self.config.n_cfm_steps,
                    )
                    self.logger.log_sampled(sampled, step=self.step)

                # Validation step
                if self.step % self.valid_steps == 0:
                    self.validation(valid_dloader=self.valid_dloader, step=self.step)

                # Checkpoint step
                if self.step % self.checkpoint_steps == 0 and self.step > 0:
                    self.checkpoint_manager.save(
                        step=self.step,
                        model=self.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        best_loss=self.best_loss,
                    )

                # Process previous batch on denoiser_stream
                if codec_features is not None:
                    with torch.cuda.stream(self.denoiser_stream):
                        self.model.train()
                        if hasattr(self.optimizer, "train"):
                            self.optimizer.train()

                        self.denoiser_stream.wait_stream(self.codec_stream)
                        metrics = self.process_batch(codec_features)
                        self.logger.log_metrics(metrics, self.step, prefix="train")
                    codec_features.record_stream(self.denoiser_stream)

                # Prepare next batch on codec_stream
                with torch.cuda.stream(self.codec_stream):
                    codec_features = self.prepare_batch(batch)

                # Update progress and step
                self.logger.update_progress(1)
                self.state.step += 1

                if self.step >= self.max_steps:
                    self.logger.close()
                    return

            # Process last batch if needed
            if self.step < self.max_steps and codec_features is not None:
                with torch.cuda.stream(self.denoiser_stream):
                    self.denoiser_stream.wait_stream(self.codec_stream)

                    if hasattr(self.optimizer, "train"):
                        self.optimizer.train()
                    self.model.train()

                    metrics = self.process_batch(codec_features)
                    self.logger.log_metrics(metrics, self.step, prefix="train")

                self.denoiser_stream.synchronize()
                self.logger.update_progress(1)
                self.state.step += 1

    # These overrides just add stream syncs on top of the base behavior

    @torch.inference_mode()
    def _sampling_step(self):
        self.denoiser_stream.synchronize()
        super()._sampling_step()

    def _validation_step(self):
        self.denoiser_stream.synchronize()
        super()._validation_step()

    def _checkpoint_step(self):
        self.denoiser_stream.synchronize()
        super()._checkpoint_step()


class ProfilingTwoStreamTrainer(ProfilingTrainer):
    """Two-stream profiling trainer with overlapped codec and denoiser work.

    Processes training step on denoiser_stream while preparing next batch
    on codec_stream for maximum GPU utilization.
    """

    def __init__(
        self,
        config: TrainerConfig,
        codec: Codec,
        model: ConditionalFlowMatcher,
        t_scheduler: TimestepScheduler,
        optimizer: Optimizer,
        scaler: GradScaler | None,
        train_dloader: DataLoader,
        train_augments: Augmentation,
        state: TrainerState | None = None,
    ):
        super().__init__(
            config=config,
            codec=codec,
            model=model,
            t_scheduler=t_scheduler,
            optimizer=optimizer,
            scaler=scaler,
            train_dloader=train_dloader,
            train_augments=train_augments,
            state=state,
        )
        self.codec_stream = torch.cuda.Stream(device=model.device)
        self.denoiser_stream = torch.cuda.current_stream(device=model.device)

    def train(self):
        """Minimal training loop with two-stream overlap."""
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()
        self.model.train()

        codec_features = None

        while self.step < self.max_steps:
            for batch in self.train_dloader:
                # Train on previous batch while codec prepares next
                if codec_features is not None:
                    torch.cuda.nvtx.range_push("launch_train")
                    with torch.cuda.stream(self.denoiser_stream):
                        self.denoiser_stream.wait_stream(self.codec_stream)
                        self.process_batch(codec_features)
                    torch.cuda.nvtx.range_pop()  # launch_train
                    codec_features.record_stream(self.denoiser_stream)

                # Prepare current batch (launches async)
                torch.cuda.nvtx.range_push("launch_codec_prep")
                with torch.cuda.stream(self.codec_stream):
                    codec_features = self.prepare_batch(batch)
                torch.cuda.nvtx.range_pop()  # launch_codec_prep

                # Sync denoiser to ensure step completes before counting
                self.state.step += 1

                if self.step >= self.max_steps:
                    return

            # Handle last batch if we haven't hit max_steps
            if self.step < self.max_steps and codec_features is not None:
                torch.cuda.nvtx.range_push("launch_train")
                with torch.cuda.stream(self.denoiser_stream):
                    self.denoiser_stream.wait_stream(self.codec_stream)
                    self.process_batch(codec_features)
                torch.cuda.nvtx.range_pop()  # launch_train

                self.denoiser_stream.synchronize()
                self.state.step += 1
