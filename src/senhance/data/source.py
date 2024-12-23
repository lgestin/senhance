import json
from pathlib import Path

import pyarrow as pa
import torch

from senhance.data.audio import Audio, AudioInfo


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
        if self.sequence_length_s is not None:
            generator = torch.Generator().manual_seed(idx)
            audio = audio.salient_excerpt(
                duration_s=self.sequence_length_s,
                generator=generator,
            )
        return audio


class ArrowAudioSource:
    def __init__(self, arrow_file: str, sequence_length_s: float = None):
        if not isinstance(arrow_file, Path):
            arrow_file = Path(arrow_file)
        self.arrow_file = arrow_file
        self.sequence_length_s = sequence_length_s

        with pa.memory_map(arrow_file.as_posix(), "rb") as arrow:
            source = pa.ipc.open_file(arrow).read_all()
        self.source = source

    def __len__(self):
        return self.source.num_rows

    def __getitem__(self, idx: int) -> Audio:
        item = self.source.slice(idx, 1)
        filepath = item["filepath"].to_pylist()[0]
        filepath = (self.arrow_file.parent / filepath).as_posix()
        waveform = item["waveform"].to_numpy()[0]
        waveform = torch.from_numpy(waveform[None]).float()
        sample_rate = int(item["sample_rate"].to_pylist()[0])

        audio = Audio(
            filepath=filepath, waveform=waveform, sample_rate=sample_rate
        )

        if self.sequence_length_s is not None:
            generator = torch.Generator().manual_seed(idx)
            audio = audio.salient_excerpt(
                duration_s=self.sequence_length_s,
                generator=generator,
            )
        return audio


if __name__ == "__main__":
    # asource = AudioSource("/data/denoising/speech/daps/clean/index.json")
    asource = ArrowAudioSource(
        "/data/denoising/speech/DNS4/datasets_fullband/clean_fullband/datasets_fullband/clean_fullband/data.test.arrow"
    )
