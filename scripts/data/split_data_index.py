import json
import random
from pathlib import Path


def split_data_index(
    data_index_path: str,
    n_train: int | float,
    n_valid: int | float,
    n_test: int | float,
    min_duration_s: float = 0.0,
):
    with open(data_index_path, "r") as f:
        index = json.load(f)
    index = [idx for idx in index if idx["duration_s"] >= min_duration_s]
    random.shuffle(index)

    n_total = len(index)
    splits = {"train": n_train, "valid": n_valid, "test": n_test}

    # convert ratio to n
    if all(isinstance(n, float) for n in splits):
        assert sum(splits.values()) == 1.0
        for key, ratio in splits.copy().items():
            splits[key] = int(ratio * n_total)
        while sum(splits.values()) < n_total:
            splits[list(splits.keys)[0]] += 1

    assert sum(splits.values()) == n_total
    splits_index = {}
    i = 0
    for key, n in splits.items():
        splits_index[key] = index[i : i + n]
        i += n

    data_index_path = Path(data_index_path)
    for split, index in splits_index.items():
        split_index_path = (
            data_index_path.parent / f"{data_index_path.stem}.{split}.json"
        )
        with open(split_index_path, "w") as f:
            json.dump(index, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_index_path", type=str, required=True)
    parser.add_argument("--n_train", type=int, required=True)
    parser.add_argument("--n_valid", type=int, required=True)
    parser.add_argument("--n_test", type=int, required=True)
    parser.add_argument("--min_duration_s", type=float, default=0.0)

    options = parser.parse_args()

    split_data_index(**vars(options))
