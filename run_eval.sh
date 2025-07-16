#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -o, --output-folder FOLDER    Output folder (required)"
    echo "  -d, --redis-db DB             Redis database number (required)"
    echo "  -p, --redis-prefix PREFIX     Redis prefix (required)"
    echo "  -e, --conda-env ENV           Conda environment name (required)"
    echo "  -h, --help                    Display this help message"
    echo ""
    echo "Example:"
    echo "  $0 -o my_results -d 15 -p my_evolution -e my_env"
    echo "  $0 --output-folder my_results --redis-db 15 --redis-prefix my_evolution --conda-env my_env"
}

# Parse command line arguments
OUTPUT_FOLDER=""
REDIS_DB=""
REDIS_PREFIX=""
CONDA_ENV=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        -d|--redis-db)
            REDIS_DB="$2"
            shift 2
            ;;
        -p|--redis-prefix)
            REDIS_PREFIX="$2"
            shift 2
            ;;
        -e|--conda-env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Check if all required arguments are provided
if [[ -z "$OUTPUT_FOLDER" || -z "$REDIS_DB" || -z "$REDIS_PREFIX" || -z "$CONDA_ENV" ]]; then
    echo "Error: All arguments are required."
    echo ""
    usage
    exit 1
fi

# Display configuration
echo "Configuration:"
echo "  Output folder: $OUTPUT_FOLDER"
echo "  Redis DB: $REDIS_DB"
echo "  Redis prefix: $REDIS_PREFIX"
echo "  Conda environment: $CONDA_ENV"
echo ""

source /home/user/conda/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

while true; do
  echo "[$(date)] Running evolution_fitness_analyzer..."
  python evolution_fitness_analyzer.py --output-folder "$OUTPUT_FOLDER" --redis-db "$REDIS_DB" --redis-prefix "$REDIS_PREFIX"
  sleep 300
done