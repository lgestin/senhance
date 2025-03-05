from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from senhance.models.checkpoint import Checkpoint
from senhance.models.unet.unet import UNET1d, UNET1dDims


@pytest.fixture
def dims():
    dims = UNET1dDims(in_dim=8, dim=16, t_dim=4)
    return dims


@pytest.fixture
def model(dims):
    model = UNET1d(dims)
    return model


@pytest.fixture
def opt(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return opt


@pytest.fixture
def scaler():
    scaler = torch.amp.GradScaler()
    return scaler


def test_checkpoint(dims, model, opt, scaler):
    with TemporaryDirectory() as tdir:
        checkpoint_path = Path(tdir) / "checkpoint.pt"
        checkpoint = Checkpoint(
            codec="dac",
            step=0,
            best_loss=0.0,
            dims=dims,
            model=model,
            opt=opt,
            scaler=scaler,
        )
        checkpoint.save(checkpoint_path)
        checkpoint.executor.shutdown()

        loaded = Checkpoint.load(checkpoint_path)

        assert checkpoint == loaded
