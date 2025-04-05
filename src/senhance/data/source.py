import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from senhance.data.audio import Audio, AudioInfo


class IndexAudioSource:
    def __init__(self, index_file: str, sequence_length_s: float = None):
        """
        index_file: str Path to a source file. A source file is a json.
        """

        self.index_file = Path(index_file)
        self.sequence_length_s = sequence_length_s

        with open(index_file, "r") as f:
            index = json.load(f)
        self.index = index

        indices = range(len(self.index))
        if sequence_length_s:
            durations_s = [item["duration_s"] for item in index]
            indices = filter(lambda i: durations_s[i] >= sequence_length_s, indices)
        self.indices = list(indices)
        if len(self.indices) == 0:
            raise ValueError(
                f"No valid samples in {self.index_file} with sequence_length_s={self.sequence_length_s}"
            )

    def __len__(self):
        return 10_000_000  # approx inifinite length

    def __getitem__(self, idx: int) -> Audio:
        source_idx = self.indices[idx % len(self.indices)]
        audioinfo = self.index[source_idx]
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
    def __init__(
        self,
        arrow_file: str,
        sequence_length_s: float = None,
        is_speech: bool = True,
    ):
        if not isinstance(arrow_file, Path):
            arrow_file = Path(arrow_file)
        self.arrow_file = arrow_file
        self.sequence_length_s = sequence_length_s

        self.memory_map = pa.memory_map(arrow_file.as_posix(), "rb")
        self.source = pa.ipc.open_file(self.memory_map)
        self.num_chunks = self.source.num_record_batches
        self.chunk_size = 128

        valid_indices = []
        for chunk_id in range(self.num_chunks):
            chunk = self.source.get_record_batch(chunk_id)
            local_indices = np.arange(chunk.num_rows, dtype=np.int32)
            if sequence_length_s:
                duration_s = chunk["duration_s"].to_numpy()
                local_indices = local_indices[duration_s >= sequence_length_s + 1e-2]
            # Convert local indices to global indices
            global_indices = local_indices + (chunk_id * self.chunk_size)
            valid_indices.extend(global_indices.tolist())
        self.indices = valid_indices
        self.is_speech = is_speech

    def __len__(self):
        return 10_000_000  # approx inifinite length

    def __getitem__(self, idx: int) -> Audio:
        source_idx = self.indices[idx % len(self.indices)]
        chunk_id = source_idx // self.chunk_size
        row = source_idx % self.chunk_size

        chunk = self.source.get_record_batch(chunk_id)
        filepath = chunk["filepath"][row].as_py()
        filepath = (self.arrow_file.parent / filepath).as_posix()
        waveform_bytes = chunk["waveform_i16"][row].as_buffer()
        count = chunk["num_samples"][row].as_py()
        sample_rate = chunk["sample_rate"][row].as_py()
        loudness = chunk["loudness"][row].as_py()
        waveform = (
            torch.frombuffer(memoryview(waveform_bytes), dtype=torch.int16, count=count)
            .view(1, -1)
            .to(dtype=torch.float)
            .div_(32768)
        )
        audio = Audio(
            filepath=filepath,
            waveform=waveform,
            sample_rate=sample_rate,
            loudness=loudness,
        )
        if self.sequence_length_s:
            generator = torch.Generator().manual_seed(idx)
            if self.is_speech:
                audio = audio.salient_excerpt(
                    duration_s=self.sequence_length_s,
                    generator=generator,
                )
            else:
                audio = audio.random_excerpt(
                    duration_s=self.sequence_length_s,
                    generator=generator,
                )

        # Load encoded latents if available
        encoded = None
        if "codec_shape" in chunk.schema.names:
            codec_shape = chunk["codec_shape"][row].as_py()
            codec_bytes = chunk["codec_bytes"][row].as_buffer()
            encoded = torch.frombuffer(memoryview(codec_bytes), dtype=torch.float16)
            encoded = encoded.view(*codec_shape)

        return audio


if __name__ == "__main__":
    # asource = AudioSource("/data/denoising/speech/daps/clean/index.json")
    asource = ArrowAudioSource(
        "/data/denoising/speech/DNS4/datasets_fullband/clean_fullband/datasets_fullband/clean_fullband/data.test.arrow"
    )
