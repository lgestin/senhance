"""Utilities for loading augmentation configurations from YAML files."""

from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_augmentation_from_yaml(
    config_path: str | Path,
    sequence_length_s: float,
    split: str,
):
    """Load augmentation from YAML configuration using Hydra.

    Args:
        config_path: Path to the YAML config file
        sequence_length_s: Sequence length in seconds
        split: Dataset split ('train', 'valid', 'test')

    Returns:
        Instantiated augmentation object

    Note:
        noise_folder and augmentation_p must be defined in the YAML config file.

    Example:
        >>> aug = load_augmentation_from_yaml(
        ...     config_path="configs/augmentation/default.yaml",
        ...     sequence_length_s=1.0,
        ...     split="train",
        ... )
    """
    config_path = Path(config_path).absolute()

    if not config_path.exists():
        raise ValueError(f"Config file does not exist: {config_path}")

    config_dir = config_path.parent
    config_name = config_path.stem  # Filename without extension

    # Compute sequence length with padding for background noise/reverb
    sequence_length_s_padded = sequence_length_s + 0.1

    # Initialize Hydra with absolute config path
    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        # Build overrides list - always add runtime parameters
        overrides = [
            f"+sequence_length_s_padded={sequence_length_s_padded}",
            f"+split={split}",
        ]

        # Compose config with overrides
        cfg = compose(
            config_name=config_name,
            overrides=overrides,
        )

        # Instantiate the augmentation object (nested under 'augmentation' key)
        augmentation = instantiate(cfg.augmentation)

    return augmentation


def load_augmentation_from_dict(
    config_dict: dict[str, Any],
    **overrides: Any,
):
    """Load augmentation from a dictionary configuration.

    Useful for programmatic config generation or testing.

    Args:
        config_dict: Dictionary containing augmentation configuration
        **overrides: Additional overrides to merge into config

    Returns:
        Instantiated augmentation object
    """
    cfg = OmegaConf.create(config_dict)

    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    return instantiate(cfg)
