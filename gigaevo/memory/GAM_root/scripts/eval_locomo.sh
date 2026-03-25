#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env

# Set output directory
outputdir=./results/locomo

# Create output directory
mkdir -p $outputdir

# Run LoCoMo evaluation
python3 eval/locomo_test.py \
    --data ./data/locomo/locomo10.json \
    --outdir $outputdir \
    --start-idx 0 \
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
    --working-api-type "openai"