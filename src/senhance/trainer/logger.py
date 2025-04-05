import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text as RichText
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from senhance.trainer.sampler import Sampled


class Logger(ABC):
    """Base class for all loggers."""

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        pass

    @abstractmethod
    def log_audio(self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int):
        """Log audio waveform."""
        pass

    @abstractmethod
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image."""
        pass

    @abstractmethod
    def log_video(self, tag: str, video: torch.Tensor, step: int, fps: int = 10):
        """Log a video."""
        pass

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "train"):
        """Log multiple metrics at once."""
        pass

    @abstractmethod
    def set_description(self, description: str):
        """Set current status description (for progress bars)."""
        pass

    @abstractmethod
    def update_progress(self, n: int = 1):
        """Update progress counter."""
        pass

    @abstractmethod
    def close(self):
        """Close the logger and clean up resources."""
        pass

    def log_waveform_sequence_as_video(
        self, tag: str, waveforms: torch.Tensor, step: int, fps: int = 10
    ):
        """Log a sequence of waveforms as a video (usually mel spectrograms).

        This is optional - loggers can implement this if they support it.
        Default implementation does nothing.
        """
        pass

    def log_sampled(self, sampled: Sampled, step: int):
        for i in range(sampled.batch_size):
            self.log_audio(
                f"{i}/cleaned.wav",
                sampled.cleaned[i],
                step=step,
                sample_rate=sampled.sample_rate,
            )
            self.log_waveform_sequence_as_video(
                f"{i}/cleaned_video.wav", sampled.cleaned_sequence[i], step=step, fps=10
            )
            if step == 0:
                self.log_audio(
                    f"{i}/noisy.wav",
                    sampled.noisy[i],
                    step=step,
                    sample_rate=sampled.sample_rate,
                )


class TensorBoardLogger(Logger):
    """TensorBoard logger implementation."""

    def __init__(
        self, log_dir: str | Path, mel_spectrogram=None, sample_rate: int = 24000
    ):
        self.writer = SummaryWriter(log_dir)
        self.current_step = 0
        self.mel_spectrogram = mel_spectrogram
        self.sample_rate = sample_rate

    def _normalize_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to normalized mel spectrogram.

        Args:
            waveform: Audio waveform tensor

        Returns:
            Normalized mel spectrogram in range [0, 1]
        """
        mels = self.mel_spectrogram(waveform[None]).log()
        mels = mels.flip(1)
        # Fixed scale: typical log mel range is roughly [-11, 2]
        mels = (mels + 11.5) / 13.5
        mels = mels.clamp(0, 1)
        return mels

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)
        self.current_step = step

    def log_audio(self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int):
        """Log audio waveform to TensorBoard."""
        # Match train.py format: append .wav if not present
        audio_tag = f"{tag}.wav" if not tag.endswith(".wav") else tag
        self.writer.add_audio(audio_tag, waveform, step, sample_rate)

        # Also log mel spectrogram if available
        if self.mel_spectrogram is not None:
            mels = self._normalize_mel_spectrogram(waveform)
            # Match train.py format: use same tag but with .png extension
            mel_tag = tag.replace(".wav", "") if tag.endswith(".wav") else tag
            mel_tag = f"{mel_tag}.png"
            self.writer.add_image(mel_tag, mels, step)

    def log_waveform_sequence_as_video(
        self, tag: str, waveforms: torch.Tensor, step: int, fps: int = 10
    ):
        """Log a sequence of waveforms as a mel spectrogram video.

        Args:
            tag: Tag for the video
            waveforms: Tensor of shape [timesteps, channels, samples]
            step: Current training step
            fps: Frames per second for the video
        """
        if self.mel_spectrogram is None:
            # If no mel spectrogram, fall back to regular video logging
            # (though this doesn't make sense for waveforms)
            return

        # Process each timestep into a mel spectrogram
        mels_frames = []
        for t_idx in range(waveforms.shape[0]):
            waveform_t = waveforms[t_idx]
            mels = self._normalize_mel_spectrogram(waveform_t)
            # Convert grayscale to RGB by repeating channels
            mels_rgb = mels.repeat(3, 1, 1)
            mels_frames.append(mels_rgb)

        # Stack frames: [timestep, channels, height, width]
        video = torch.stack(mels_frames, dim=0)
        # Add batch dimension: [batch=1, timestep, channels, height, width]
        self.writer.add_video(tag, video[None], step, fps=fps)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image to TensorBoard."""
        self.writer.add_image(tag, image, step)

    def log_video(self, tag: str, video: torch.Tensor, step: int, fps: int = 10):
        """Log a video to TensorBoard."""
        self.writer.add_video(tag, video, step, fps=fps)

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "train"):
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_scalar(f"{prefix}/{key}", value, step)

    def set_description(self, description: str):
        """TensorBoard doesn't use descriptions, so this is a no-op."""
        pass

    def update_progress(self, n: int = 1):
        """TensorBoard doesn't track progress, so this is a no-op."""
        pass

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class StdoutLogger(Logger):
    """Stdout logger implementation using tqdm for progress bars."""

    def __init__(
        self,
        total_steps: int,
        initial_step: int = 0,
        unit: str = "batch",
        smoothing: float = 0.1,
    ):
        self.pbar = tqdm(
            initial=initial_step,
            total=total_steps,
            unit=unit,
            smoothing=smoothing,
        )
        self.current_metrics = {}
        self.current_step = initial_step

    def log_scalar(self, tag: str, value: float, step: int):
        """Store scalar for display in progress bar."""
        self.current_metrics[tag] = value
        self.current_step = step

    def log_audio(self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int):
        """Stdout doesn't log audio, so this is a no-op."""
        pass

    def log_waveform_sequence_as_video(
        self, tag: str, waveforms: torch.Tensor, step: int, fps: int = 10
    ):
        """Stdout doesn't log videos, so this is a no-op."""
        pass

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Stdout doesn't log images, so this is a no-op."""
        pass

    def log_video(self, tag: str, video: torch.Tensor, step: int, fps: int = 10):
        """Stdout doesn't log videos, so this is a no-op."""
        pass

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "train"):
        """Update current metrics and refresh progress bar."""
        for key, value in metrics.items():
            self.current_metrics[f"{prefix}/{key}"] = value
        self.current_step = step
        self._update_description()

    def set_description(self, description: str):
        """Set the progress bar description."""
        self.pbar.set_description_str(description)

    def update_progress(self, n: int = 1):
        """Update the progress bar."""
        self.pbar.update(n)

    def _update_description(self):
        """Update description with current metrics."""
        if self.current_metrics:
            # Find the most important metric (loss if available)
            loss_key = None
            for key in ["train/loss", "valid/loss", "loss"]:
                if key in self.current_metrics:
                    loss_key = key
                    break

            if loss_key:
                prefix = loss_key.split("/")[0].upper() if "/" in loss_key else "TRAIN"
                desc = f"{prefix} {self.current_step} | {self.current_metrics[loss_key]:.4f}"
                self.set_description(desc)

    def close(self):
        """Close the progress bar."""
        self.pbar.close()


class SpeedColumn(ProgressColumn):
    """Custom column to display speed with proper None handling."""

    def render(self, task: Task) -> RichText:
        """Render the speed."""
        speed = task.finished_speed if task.finished else task.speed
        if speed is None:
            return RichText("  --.-- it/s", style="cyan")
        unit = task.fields.get("unit", "it")
        return RichText(f"{speed:>6.2f} {unit}/s", style="cyan")


class RichLogger(Logger):
    """Rich-based logger with live updating display."""

    def __init__(
        self,
        total_steps: int,
        initial_step: int = 0,
        unit: str = "batch",
        refresh_per_second: int = 4,
        config: Any = None,
    ):
        self.console = Console()
        self.total_steps = total_steps
        self.current_step = initial_step
        self.unit = unit
        self.config = config

        # Timing (must be set early as it's used in _create_layout)
        self.start_time = datetime.now()

        # Metrics storage
        self.train_metrics: dict[str, float] = {}
        self.valid_metrics: dict[str, float] = {}
        self.current_phase = "train"

        # Progress tracking
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("["),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.fields[unit]}"),
            TextColumn("]"),
            SpeedColumn(),
            TimeElapsedColumn(),
            TextColumn("/"),
            TimeRemainingColumn(),
            console=self.console,
        )

        # Create main progress task
        self.task_id = self.progress.add_task(
            "[cyan]Training",
            total=total_steps,
            completed=initial_step,
            unit=unit,
        )

        # Live display
        self.live = Live(
            self._create_layout(),
            console=self.console,
            refresh_per_second=refresh_per_second,
        )
        self.live.start()

    def _create_layout(self) -> Layout:
        """Create the rich layout for display."""
        layout = Layout()

        if self.config is not None:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="command", size=5),
                Layout(name="config", size=10),
                Layout(name="progress", size=3),
                Layout(name="metrics", size=12),
            )
        else:
            layout.split_column(
                Layout(name="header", size=3),
                Layout(name="progress", size=3),
                Layout(name="metrics", size=12),
            )

        # Header
        header_text = RichText("Senhance Training", style="bold cyan", justify="center")
        layout["header"].update(Panel(header_text, border_style="cyan"))

        # Command panel (if config provided)
        if self.config is not None:
            command_text = " ".join(sys.argv)
            layout["command"].update(
                Panel(
                    RichText(command_text, style="dim white", overflow="fold"),
                    title="[bold]Command",
                    border_style="green",
                )
            )

        # Config panel (if provided)
        if self.config is not None:
            layout["config"].update(self._create_config_panel())

        # Progress bar
        layout["progress"].update(self.progress)

        # Metrics table
        layout["metrics"].update(self._create_metrics_table())

        return layout

    def _create_config_panel(self) -> Panel:
        """Create a panel displaying configuration."""
        from dataclasses import fields, is_dataclass

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        if is_dataclass(self.config):
            # Handle dataclass configs
            for field in fields(self.config):
                value = getattr(self.config, field.name)
                # Format value
                if isinstance(value, (int, float)):
                    value_str = (
                        f"{value:,}" if isinstance(value, int) else f"{value:.2e}"
                    )
                else:
                    value_str = str(value)
                    # Truncate long strings
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                table.add_row(field.name, value_str)
        elif isinstance(self.config, dict):
            # Handle dict configs
            for key, value in self.config.items():
                if isinstance(value, (int, float)):
                    value_str = (
                        f"{value:,}" if isinstance(value, int) else f"{value:.2e}"
                    )
                else:
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                table.add_row(key, value_str)
        else:
            # Fallback to string representation
            table.add_row("config", str(self.config))

        return Panel(table, title="[bold]Configuration", border_style="blue")

    def _create_metrics_table(self) -> Table:
        """Create a table displaying current metrics."""
        table = Table(
            title="Training Metrics", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Train", justify="right", style="green")
        table.add_column("Valid", justify="right", style="yellow")

        # Collect all metric keys
        all_keys = set(self.train_metrics.keys()) | set(self.valid_metrics.keys())

        # Remove prefix for display
        metric_names = set()
        for key in all_keys:
            if "/" in key:
                metric_names.add(key.split("/", 1)[1])
            else:
                metric_names.add(key)

        # Sort metrics (put loss first if available)
        sorted_metrics = sorted(metric_names, key=lambda x: (x != "loss", x))

        for metric in sorted_metrics:
            train_key = f"train/{metric}"
            valid_key = f"valid/{metric}"

            train_val = self.train_metrics.get(
                train_key, self.train_metrics.get(metric)
            )
            valid_val = self.valid_metrics.get(
                valid_key, self.valid_metrics.get(metric)
            )

            train_str = f"{train_val:.6f}" if train_val is not None else "-"
            valid_str = f"{valid_val:.6f}" if valid_val is not None else "-"

            table.add_row(metric, train_str, valid_str)

        # Add step info
        table.add_row("", "", "", style="dim")
        table.add_row(
            "Step", f"{self.current_step:,}", f"{self.current_step:,}", style="bold"
        )

        # Add timing info
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split(".")[0]  # Remove microseconds
        table.add_row("Elapsed", elapsed_str, "", style="dim")

        return table

    def _update_display(self):
        """Update the live display."""
        self.live.update(self._create_layout())

    def log_scalar(self, tag: str, value: float, step: int):
        """Store scalar for display."""
        if tag.startswith("train/"):
            self.train_metrics[tag] = value
        elif tag.startswith("valid/"):
            self.valid_metrics[tag] = value
        else:
            # Default to current phase
            prefixed_tag = f"{self.current_phase}/{tag}"
            if self.current_phase == "train":
                self.train_metrics[prefixed_tag] = value
            else:
                self.valid_metrics[prefixed_tag] = value

        self.current_step = step
        self._update_display()

    def log_audio(self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int):
        """Rich logger doesn't log audio, so this is a no-op."""
        pass

    def log_waveform_sequence_as_video(
        self, tag: str, waveforms: torch.Tensor, step: int, fps: int = 10
    ):
        """Rich logger doesn't log videos, so this is a no-op."""
        pass

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Rich logger doesn't log images, so this is a no-op."""
        pass

    def log_video(self, tag: str, video: torch.Tensor, step: int, fps: int = 10):
        """Rich logger doesn't log videos, so this is a no-op."""
        pass

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "train"):
        """Log multiple metrics at once."""
        self.current_phase = prefix

        for key, value in metrics.items():
            prefixed_key = f"{prefix}/{key}"
            if prefix == "train":
                self.train_metrics[prefixed_key] = value
            else:
                self.valid_metrics[prefixed_key] = value

        self.current_step = step
        self._update_display()

    def set_description(self, description: str):
        """Set the progress bar description."""
        self.progress.update(self.task_id, description=description)
        self._update_display()

    def update_progress(self, n: int = 1):
        """Update the progress bar."""
        self.progress.update(self.task_id, advance=n)
        self._update_display()

    def close(self):
        """Close the live display."""
        self.live.stop()

        # Print final summary
        final_table = self._create_metrics_table()
        self.console.print("\n")
        self.console.print(
            Panel(final_table, title="[bold]Training Complete", border_style="green")
        )


class RichValidationLogger:
    """Rich-based validation logger with nested display."""

    def __init__(self, total: int, leave: bool = False):
        self.console = Console()
        self.total = total
        self.leave = leave
        self.metrics_accumulator = defaultdict(list)
        self.current_step = 0

        # Create progress bar
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[yellow]Validation"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
        )

        self.task_id = self.progress.add_task(
            "Validating",
            total=total,
        )

        self.progress.start()

    def update(self, metrics: dict[str, torch.Tensor], step: int):
        """Update validation metrics."""
        for key, value in metrics.items():
            self.metrics_accumulator[key].append(value)
        self.current_step = step

        # Update description with latest loss
        if "loss" in metrics:
            desc = f"[yellow]Validation[/yellow] [dim]Step {step}[/dim] Loss: {metrics['loss']:.4f}"
            self.progress.update(self.task_id, description=desc)

        self.progress.update(self.task_id, advance=1)

    def get_averaged_metrics(self) -> dict[str, float]:
        """Get averaged metrics over all validation batches."""
        averaged = {}
        for key, values in self.metrics_accumulator.items():
            averaged[key] = torch.stack(values).mean()
        return averaged

    def close(self):
        """Close the progress bar and optionally print summary."""
        self.progress.stop()

        if not self.leave:
            # Clear the progress bar
            self.console.clear()
        else:
            # Print validation summary
            avg_metrics = self.get_averaged_metrics()
            if avg_metrics:
                table = Table(title="Validation Results", show_header=True)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", justify="right", style="yellow")

                for key, value in sorted(avg_metrics.items()):
                    table.add_row(key, f"{value:.6f}")

                self.console.print(table)

        self.metrics_accumulator.clear()


class CompositeLogger(Logger):
    """Composite logger that forwards calls to multiple loggers."""

    def __init__(self, loggers: list[Logger], mel_spectrogram=None):
        self.loggers = loggers
        self.mel_spectrogram = mel_spectrogram
        # Pass mel_spectrogram to TensorBoard loggers
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger) and mel_spectrogram is not None:
                logger.mel_spectrogram = mel_spectrogram

    def log_scalar(self, tag: str, value: float, step: int):
        """Forward to all loggers."""
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)

    def log_audio(self, tag: str, waveform: torch.Tensor, step: int, sample_rate: int):
        """Forward to all loggers."""
        for logger in self.loggers:
            logger.log_audio(tag, waveform, step, sample_rate)

    def log_waveform_sequence_as_video(
        self, tag: str, waveforms: torch.Tensor, step: int, fps: int = 10
    ):
        """Forward to all loggers that support waveform sequence videos."""
        for logger in self.loggers:
            if hasattr(logger, "log_waveform_sequence_as_video"):
                logger.log_waveform_sequence_as_video(tag, waveforms, step, fps)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Forward to all loggers."""
        for logger in self.loggers:
            logger.log_image(tag, image, step)

    def log_video(self, tag: str, video: torch.Tensor, step: int, fps: int = 10):
        """Forward to all loggers."""
        for logger in self.loggers:
            logger.log_video(tag, video, step, fps)

    def log_metrics(self, metrics: dict[str, float], step: int, prefix: str = "train"):
        """Forward to all loggers."""
        for logger in self.loggers:
            logger.log_metrics(metrics, step, prefix)

    def set_description(self, description: str):
        """Forward to all loggers."""
        for logger in self.loggers:
            logger.set_description(description)

    def update_progress(self, n: int = 1):
        """Forward to all loggers."""
        for logger in self.loggers:
            logger.update_progress(n)

    def close(self):
        """Close all loggers."""
        for logger in self.loggers:
            logger.close()


class ValidationLogger:
    """Special logger for validation loops with nested progress bar."""

    def __init__(self, total: int, leave: bool = False):
        self.pbar = tqdm(total=total, leave=leave)
        self.metrics_accumulator = defaultdict(list)
        self.current_step = 0

    def update(self, metrics: dict[str, torch.Tensor], step: int):
        """Update validation metrics."""
        for key, value in metrics.items():
            self.metrics_accumulator[key].append(value)
        self.current_step = step

        # Update description with latest loss
        if "loss" in metrics:
            self.pbar.set_description_str(f"VALID {step} | {metrics['loss']:.4f}")

        self.pbar.update(1)

    def get_averaged_metrics(self) -> dict[str, float]:
        """Get averaged metrics over all validation batches."""
        averaged = {}
        for key, values in self.metrics_accumulator.items():
            averaged[key] = torch.stack(values).mean()
        return averaged

    def close(self):
        """Close the progress bar."""
        self.pbar.close()
        self.metrics_accumulator.clear()
