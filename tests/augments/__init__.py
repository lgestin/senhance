from pathlib import Path

TEST_SEED = 42
ASSETS_FOLDER = Path(__file__).parent.parent / "assets"

AUDIO_TEST_FILES = [fpath.as_posix() for fpath in ASSETS_FOLDER.glob("*.wav")]
