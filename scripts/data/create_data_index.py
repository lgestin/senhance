import json
from tqdm import tqdm
from pathlib import Path


from denoiser.data.audio import Audio
from concurrent.futures import ThreadPoolExecutor


def create_data_index(
    data_folder: str,
    index_path: str = None,
    split: str = None,
    n_workers: int = 1,
):
    data_folder = Path(data_folder)

    def extract_informations(file: Path):
        relative_path = file.relative_to(data_folder)
        audio = Audio(file.as_posix())

        info = {
            "filepath": relative_path.as_posix(),
            "sample_rate": audio.sample_rate,
            "duration": audio.duration,
            "loudness": audio.loudness,
        }
        return info

    with ThreadPoolExecutor(n_workers) as executor:
        futures = []
        for file in data_folder.glob("**/*.wav"):
            future = executor.submit(extract_informations, file)
            futures.append(future)

    index = []
    for future in tqdm(futures):
        info = future.result()
        index.append(info)

    if index_path is None:
        index_path = data_folder / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    create_data_index("/data/denoising/speech/daps", None, n_workers=8)
