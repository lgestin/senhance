import json
import torch

from pathlib import Path
from torch.utils.data import Dataset

from denoiser.data.audio import Audio


class AudioSource:
    def __init__(self, index_file: str, sequence_length: int = None):
        """
        index_file: str Path to a source file. A source file is a json.
        """

        self.index_file = Path(index_file)
        self.sequence_length = sequence_length

        with open(index_file, "r") as f:
            index = json.load(f)
        self.index = index

    def __getitem__(self, i: int):
        file = self.index[i]
        path = file.get("filepath")
        sr = file.get("sample_rate")
        duration = file.get("duration")
        n_frames = int(duration * sr)

        if self.sequence_length is None:
            start, end = 0, None
        else:
            start = torch.randint(0, n_frames - self.sequence_length, size=(1,))
            end = start + self.sequence_length
        audio = Audio(self.index_file.parent / path, start=start, end=end)
        return audio
