import json
from tqdm import tqdm
from pathlib import Path


from denoiser.data.audio import Audio
from audiotools import AudioSignal


def create_data_index(data_folder: str, index_path: str = None, split=None):
    data_folder = Path(data_folder)
    data = []

    for file in tqdm(data_folder.glob("**/*.wav")):
        relative_path = file.relative_to(data_folder)
        # audio = AudioSignal(path)
        audio = Audio(file.as_posix())
        line = {
            "filepath": relative_path.as_posix(),
            "sample_rate": audio.sample_rate,
            "duration": audio.duration,
            # "loudness": audio.loudness(),
        }
        data.append(line)

    if index_path is None:
        index_path = data_folder / "index.json"
    with open(index_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    create_data_index("/home/lucas/data/daps/clean", None)
