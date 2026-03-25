#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env

# Set output directory
base_outputdir=./results/hotpotqa

# Create output directory
mkdir -p $base_outputdir

# Run HotpotQA evaluation
for dataset in "eval_400" "eval_1600" "eval_3200"
do
    echo "Processing dataset: $dataset"
    outputdir=$base_outputdir/${dataset}

    python3 eval/hotpotqa_test.py \
        --data ./data/hotpotqa/${dataset}.json \
        --outdir $outputdir \
        --start-idx 0 \
        --max-tokens 2048 \
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
done