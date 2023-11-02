#!/bin/bash

# Arrays of datasets and models
DATASETS=("muraki2021" "connell2007")
MODELS=("ViT-B-32" "ViT-L-14-336" "ViT-H-14" "imagebind")

# Iterate over each dataset and model combination
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Running manipulation check for dataset: $dataset with model: $model"
        python3 src/manipulation_check.py --dataset "$dataset" --model "$model"
        echo "----------------------------------------------------------"
    done
done
