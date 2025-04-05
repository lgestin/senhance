import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from tqdm import tqdm

from senhance.data.utils import load_waveform, resample
from senhance.models.codec.dac import DescriptAudioCodec


def load_and_resample(filepath, target_sample_rate=None):
    waveform, sr = load_waveform(filepath)
    waveform = waveform.mean(0, dtype=waveform.dtype, keepdims=True)
    if (target_sample_rate is not None) and (sr != target_sample_rate):
        waveform = resample(waveform, sr, target_sample_rate)
        sr = target_sample_rate
    return waveform, sr


def create_arrow_from_index(
    index_path: str,
    n_workers: int,
    chunk_size: int = 128,
    sort: bool = False,
    target_sample_rate: int = None,
    max_length_s: float = None,
    codec_path: str | None = None,
):
    with open(index_path, "r") as f:
        index = json.load(f)

    if sort:
        index = sorted(index, key=lambda x: x["duration_s"], reverse=True)

    output_folder = Path(index_path).parent
    output_file = f"{Path(index_path).stem.replace('index', 'data')}"
    if target_sample_rate is not None:
        output_file = f"{output_file}.{target_sample_rate}hz"

    output_file = output_folder / f"{output_file}.arrow"

    if max_length_s is None:
        max_length_s = 1e6

    if codec_path is not None:
        device = torch.device("cuda")
        codec = DescriptAudioCodec(path=codec_path)
        codec = codec.eval()
        codec = codec.freeze()
        codec = codec.to(device, dtype=torch.bfloat16)
        codec.encode = torch.compile(codec.encode)
        target_sample_rate = codec.sample_rate

    # Define schema based on whether we're encoding
    schema_fields = [
        pa.field("filepath", type=pa.string()),
        pa.field("waveform_i16", type=pa.binary()),
        pa.field("num_samples", type=pa.int32()),
        pa.field("sample_rate", type=pa.int32()),
        pa.field("duration_s", type=pa.float32()),
        pa.field("loudness", type=pa.float32()),
    ]

    if codec_path is not None:
        # Add encoded field for codec latents
        schema_fields.append(pa.field("codec_bytes", type=pa.binary()))

    schema = pa.schema(schema_fields)
    type_map = {f.name: f.type for f in schema}

    with pa.OSFile(output_file.as_posix(), "wb") as sink, pa.ipc.new_file(
        sink, schema
    ) as writer:
        with ThreadPoolExecutor(n_workers) as executor:
            futures = []
            for item in index:
                filepath = item["filepath"]
                future = executor.submit(
                    load_and_resample, output_folder / filepath, target_sample_rate
                )
                futures.append(future)

            results = defaultdict(list)
            for i, (item, future) in tqdm(
                enumerate(zip(index, futures)), total=len(futures)
            ):
                waveform, sr = future.result()
                for s in range(0, waveform.shape[-1], int(max_length_s * sr)):
                    chunk = np.ascontiguousarray(
                        waveform[..., s : s + int(max_length_s * sr)]
                    )
                    if codec_path is not None and chunk.size < 640:
                        continue
                    results["filepath"].append(item["filepath"])
                    results["waveform_i16"].append(chunk.tobytes())
                    results["num_samples"].append(chunk.size)
                    results["sample_rate"].append(sr)
                    duration_s = chunk.size / sr
                    results["duration_s"].append(duration_s)
                    results["loudness"].append(item["loudness"])

                    if codec_path is not None:
                        with torch.inference_mode():
                            encoded = (
                                torch.from_numpy(chunk.copy())[None]
                                .float()
                                .div_(32768.0)
                                .to(device, dtype=torch.bfloat16)
                            )
                            encoded = codec.encode(encoded).squeeze(0)
                            import ipdb

                            ipdb.set_trace()
                            results["codec_bytes"].append(
                                np.ascontiguousarray(encoded.cpu().numpy()).tobytes()
                            )

                    if len(results["waveform_i16"]) % chunk_size == 0:
                        results = {
                            k: pa.array(v, type=type_map[k]) for k, v in results.items()
                        }

                        batch = pa.record_batch(results)
                        writer.write_batch(batch)
                        results = defaultdict(list)

            results = {k: pa.array(v, type=type_map[k]) for k, v in results.items()}
            batch = pa.record_batch(results)
            writer.write_batch(batch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--sort", action="store_true")
    parser.add_argument(
        "--target_sample_rate",
        type=int,
        default=None,
        help="Resample all audio to this sample rate (e.g., 24000)",
    )
    parser.add_argument("--max_length_s", type=float, default=None)
    parser.add_argument("--codec_path", type=str, default=None)

    options = parser.parse_args()

    create_arrow_from_index(**vars(options))
