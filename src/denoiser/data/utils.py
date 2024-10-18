import torch
import torchaudio.functional as F
import soundfile as sf


def load_audio(path: str, start: int = 0, end: int = None):
    num_frames = -1 if end is None else end - start
    audio, sr = sf.read(
        path,
        start=start,
        frames=num_frames,
        dtype="float32",
        always_2d=True,
    )
    audio = torch.from_numpy(audio.T)
    return audio, sr


def resample(audio: torch.Tensor, orig_sr: int, targ_sr: int):
    return F.resample(waveform=audio, orig_freq=orig_sr, new_freq=targ_sr)


def truncated_normal(size: tuple, min_val: float, max_val: float) -> torch.Tensor:
    """
    approximation of a truncated normal distribution between min and max
    """
    normal = torch.randn(size)
    trunc_normal = torch.fmod(normal, 2)
    trunc_normal = (trunc_normal / 2 + 1) / 2
    trunc_normal = trunc_normal * (max_val - min_val)
    trunc_normal = trunc_normal + min_val
    return trunc_normal
