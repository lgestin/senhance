import json
from pathlib import Path

import torch

from denoiser.data.audio import Audio, AudioInfo


class AudioSource:
    def __init__(self, index_file: str, sequence_length_s: float = None):
        """
        index_file: str Path to a source file. A source file is a json.
        """

        self.index_file = Path(index_file)
        self.sequence_length_s = sequence_length_s

        with open(index_file, "r") as f:
            index = json.load(f)
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Audio:
        audioinfo = self.index[idx]
        audioinfo["filepath"] = self.index_file.parent / audioinfo["filepath"]
        audioinfo = AudioInfo(**audioinfo)
        audio = Audio.from_audioinfo(audioinfo)
        generator = torch.Generator().manual_seed(idx)
        if self.sequence_length_s is not None:
            audio = audio.salient_excerpt(
                duration_s=self.sequence_length_s,
                generator=generator,
            )
        return audio


if __name__ == "__main__":
    asource = AudioSource("../../data/daps/clean/index.json")
