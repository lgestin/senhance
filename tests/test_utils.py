from pathlib import Path

import numpy as np
import pytest

from senhance.data.utils import load_audio, resample

test_file_path = Path(__file__).parent / "assets/physicsworks.wav"


def test_load_audio_dtype():
    loaded, sr = load_audio(test_file_path.as_posix())
    assert loaded.dtype == np.int16
    assert isinstance(loaded, np.ndarray)
    assert loaded.ndim == 2


srs = [4000, 8000, 11025, 16000, 22050, 24000, 44100, 48000]


@pytest.mark.parametrize("orig_sr", srs)
@pytest.mark.parametrize("targ_sr", srs)
def test_resample(orig_sr, targ_sr):
    np.random.seed(0)
    t_s = 2
    waveform = np.random.randint(
        -(2**15), 2**15, (1, t_s * orig_sr), dtype=np.int16
    )
    resampled = resample(waveform, orig_sr=orig_sr, targ_sr=targ_sr)
    assert t_s * targ_sr == resampled.shape[-1]
    assert resampled.dtype == waveform.dtype
    assert waveform.shape[0] == resampled.shape[0]
