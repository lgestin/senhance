import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from senhance.data.audio import Audio, AudioInfo


def create_data_index(
    data_folder: str,
    n_workers: int = 1,
    min_duration_s: float = 0.0,
):
    data_folder = Path(data_folder)

    def extract_informations(file: Path):
        relative_path = file.relative_to(data_folder)
        audio = Audio(file.as_posix())

        info = None
        if audio.duration_s >= min_duration_s:
            info = AudioInfo(
                filepath=relative_path.as_posix(),
                sample_rate=audio.sample_rate,
                duration_s=audio.duration_s,
                loudness=audio.mono().loudness,
            )
        return info

    with ThreadPoolExecutor(n_workers) as executor:
        files = list(data_folder.rglob("*.wav"))
        infos = executor.map(extract_informations, files)
        index = list(tqdm(infos, total=len(files)))
    index = [asdict(i) for i in index if i is not None]

    index_path = data_folder / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--min_duration_s", type=float, default=0.5)

    options = parser.parse_args()
    create_data_index(**vars(options))
