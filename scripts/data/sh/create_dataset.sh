#!/bin/bash

export DATA_PATH=$1
echo $DATA_PATH

hatch run python scripts/data/create_data_index.py --data_folder $DATA_PATH --n_workers=8 --min_duration_s 0.86
hatch run python scripts/data/split_data_index.py --data_index_path $DATA_PATH/index.json --n_train 0.85 --n_valid 0.1 --n_test 0.05
for split in train valid test; do hatch run python scripts/data/create_arrow_from_index.py --index_path $DATA_PATH/index.${split}.json --n_workers 8 --chunk_size 1024; done
