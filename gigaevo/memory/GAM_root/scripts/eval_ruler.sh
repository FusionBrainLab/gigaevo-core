#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env

# Set base output directory
base_outputdir=./results/ruler

# Create base output directory
mkdir -p $base_outputdir

# Process all RULER datasets
for dataset in "qa_1" "qa_2" "vt" "niah_single_1" "niah_single_2" "niah_single_3" "niah_multikey_1" "niah_multikey_2" "niah_multikey_3" "niah_multiquery" "niah_multivalue" "cwe" "fwe"
do 
    echo "Processing dataset: $dataset"
    outputdir=$base_outputdir/${dataset}
    
    python3 eval/ruler_test.py \
        --data ./data/ruler/${dataset}.jsonl \
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

