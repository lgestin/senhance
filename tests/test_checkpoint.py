from pathlib import Path
from tempfile import TemporaryDirectory

from senhance.models.checkpoint import Checkpoint


def test_checkpoint():
    with TemporaryDirectory() as tdir:
        checkpoint_path = Path(tdir) / "checkpoint.pt"
        checkpoint = Checkpoint(
            codec="dac",
            step=0,
            best_loss=0.0,
            dims={},
            model={},
            opt={},
            scaler={},
        )
        checkpoint.save(checkpoint_path)
        checkpoint.executor.shutdown()

        loaded = Checkpoint.load(checkpoint_path)

        assert checkpoint == loaded
