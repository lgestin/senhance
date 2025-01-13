import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from senhance.data.utils import load_audio


def create_arrow_from_index(
    index_path: str,
    n_workers: int,
    chunk_size: int = 128,
    sort: bool = False,
):
    with open(index_path, "r") as f:
        index = json.load(f)

    if sort:
        index = sorted(index, key=lambda x: x["duration_s"], reverse=True)

    output_folder = Path(index_path).parent
    output_file = f"{Path(index_path).stem.replace('index', 'data')}.arrow"
    output_file = output_folder / output_file

    schema = pa.schema(
        [
            pa.field("filepath", type=pa.string()),
            pa.field("waveform", type=pa.list_(pa.int16())),
            pa.field("sample_rate", type=pa.float64()),
            pa.field("duration_s", type=pa.float64()),
            pa.field("loudness", type=pa.float64()),
        ]
    )
    with pa.OSFile(output_file.as_posix(), "wb") as sink:
        with pa.ipc.new_file(sink, schema) as writer:
            for i in tqdm(range(0, len(index), chunk_size)):
                chunk = index[i : i + chunk_size]
                table = pd.DataFrame(chunk)
                table = table.set_index("filepath")

                jobs = {}
                waveforms = defaultdict(list)
                with ThreadPoolExecutor(n_workers) as executor:
                    for filepath in table.index:
                        jobs[filepath] = executor.submit(
                            load_audio, output_folder / filepath
                        )

                    for filepath, job in jobs.items():
                        waveforms["filepath"].append(filepath)
                        waveform = job.result()[0]
                        waveform = waveform.mean(0, dtype=waveform.dtype)
                        waveforms["waveform"].append(waveform)
                        for key, val in table.loc[filepath].to_dict().items():
                            waveforms[key].append(val)

                waveforms = {k: pa.array(v) for k, v in waveforms.items()}
                batch = pa.record_batch(waveforms)
                writer.write_batch(batch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--chunk_size", type=int)
    parser.add_argument("--sort", action="store_true")

    options = parser.parse_args()
    create_arrow_from_index(**vars(options))
