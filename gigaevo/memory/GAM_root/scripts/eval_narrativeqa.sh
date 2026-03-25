#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env

# Set output directory
outputdir=./results/narrativeqa

# Create output directory
mkdir -p $outputdir

# Run NarrativeQA evaluation
python3 eval/narrativeqa_test.py \
    --data-dir ./data/narrativeqa \
    --split test \
    --outdir $outputdir \
    --start-idx 0 \
    --end-idx 300 \
    --max-tokens 2048 \
    --seed 42 \
    --memory-api-key "your-openai-api-key" \
    --memory-base-url "https://api.openai.com/v1" \
    --memory-model "gpt-4o-mini" \
    --memory-api-type "openai" \
    --research-api-key "your-openai-api-key" \
    --research-base-url "https://api.openai.com/v1" \
    --research-model "gpt-4o-mini" \
    --research-api-type "openai" \
    --working-api-key "your-openai-api-key" \
    --working-base-url "https://api.openai.com/v1" \
    --working-model "gpt-4o-mini" \
    --working-api-type "openai" \
    --embedding-model-path BAAI/bge-m3

