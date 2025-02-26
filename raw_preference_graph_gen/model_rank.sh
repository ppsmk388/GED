#!/bin/bash
# pipeline.sh
#
# This script orchestrates the full pipeline:
# 1. Generation: For each input JSON file and for each generator variant (configured via --generators),
#    run generate.py to produce output JSON files. Each file is named as <generator>_<filename>.
# 2. raw_preference_graph_gen: For each input file, compare every distinct pair of generated JSON files (for different generators)
#    by running compare.py. The compare outputs are saved with filenames of the form <gen1>_vs_<gen2>_<filename>.
# 3. Aggregation: Run aggregate.py on the compare output folder to produce a final aggregated JSON file.
#
# Usage:
#   ./pipeline.sh -i <input_path> -o <output_folder> -m <model> -g <gpu> -d <dataset> --generators <gen1,gen2,...> --api_base <api_base_url>
#                [--model_path <path>] [--port <port>] [--mode <random|output_first|reference_first>] [--max_tokens <num>] [--temperature <num>] [--threads <num>]
#
# Note: The input path is assumed to be a directory (or a single file) containing the original JSON files.
#       For each generator in the comma-separated list, generate.py will be invoked separately.

usage() {
    echo "Usage: $0 -i <input_path> -o <output_folder> -m <model> -g <gpu> -d <dataset> --generators <gen1,gen2,...> --api_base <api_base_url> [--model_path <path>] [--port <port>] [--mode <random|output_first|reference_first>] [--max_tokens <num>] [--temperature <num>] [--threads <num>]"
    exit 1
}

# Default parameters
model="local"
gpu=4
dataset=""
generators=""
mode="random"
api_base=""
port=8000
model_path=""
max_tokens=256
temperature=0.7
threads=10

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--input)
            input_path="$2"
            shift 2 ;;
        -o|--output)
            output_folder="$2"
            shift 2 ;;
        -m|--model)
            model="$2"
            shift 2 ;;
        -g|--gpu)
            gpu="$2"
            shift 2 ;;
        -d|--dataset)
            dataset="$2"
            shift 2 ;;
        --generators)
            generators="$2"
            shift 2 ;;
        --api_base)
            api_base="$2"
            shift 2 ;;
        --model_path)
            model_path="$2"
            shift 2 ;;
        --port)
            port="$2"
            shift 2 ;;
        --mode)
            mode="$2"
            shift 2 ;;
        --max_tokens)
            max_tokens="$2"
            shift 2 ;;
        --temperature)
            temperature="$2"
            shift 2 ;;
        --threads)
            threads="$2"
            shift 2 ;;
        *)
            echo "Unknown parameter: $1"
            usage ;;
    esac
done

if [ -z "$input_path" ] || [ -z "$output_folder" ] || [ -z "$api_base" ] || [ -z "$generators" ]; then
    usage
fi

# Create necessary directories
mkdir -p "$output_folder/generation"
mkdir -p "$output_folder/raw_preference_graph_gen"
mkdir -p "$output_folder/aggregation"

# # Step 1: Generation
# cd "$output_folder/generation"
# IFS=',' read -r -a gen_array <<< "$generators"
# if [ -d "$input_path" ]; then
#     for file in "$input_path"/*.json; do
#         base=$(basename "$file")
#         for gen in "${gen_array[@]}"; do
#             out_file="$output_folder/generation/${gen}_${base}"
#             echo "[INFO] Generating output for $base using generator $gen"
#             python generate.py --input_json_path "$file" --output_json_path "$out_file" --model "$model" --gpu "$gpu" --dataset "$dataset" --generator "$gen" --temperature "$temperature" --top_p 0.8 --max_tokens 2048
#         done
#     done
# else
#     base=$(basename "$input_path")
#     for gen in "${gen_array[@]}"; do
#         out_file="$output_folder/generation/${gen}_${base}"
#         echo "[INFO] Generating output for $base using generator $gen"
#         python generate.py --input_json_path "$input_path" --output_json_path "$out_file" --model "$model" --gpu "$gpu" --dataset "$dataset" --generator "$gen" --temperature "$temperature" --top_p 0.8 --max_tokens 2048
#     done
# fi

# Step 2: raw_preference_graph_gen
cd "$output_folder/raw_preference_graph_gen"
# For each original input file, compare every distinct pair of generated files.
if [ -d "$input_path" ]; then
    for file in "$input_path"/*.json; do
        base=$(basename "$file")
        # Build an array of generated files for this input file.
        gen_files=()
        for gen in "${gen_array[@]}"; do
            gen_files+=("$output_folder/generation/${gen}_${base}")
        done
        # Compare each distinct pair (i<j)
        num=${#gen_files[@]}
        for (( i=0; i<$num; i++ )); do
            for (( j=i+1; j<$num; j++ )); do
                gen1=$(basename "${gen_files[i]}" | cut -d'_' -f1)
                gen2=$(basename "${gen_files[j]}" | cut -d'_' -f1)
                comp_file="$output_folder/raw_preference_graph_gen/${gen1}_vs_${gen2}_${base}"
                echo "[INFO] Comparing ${gen1} and ${gen2} for $base"
                if [ -n "$model_path" ]; then
                    python compare.py --input_json "${gen_files[i]}" --ref_json "${gen_files[j]}" --output_json "$comp_file" --model "$model" --mode "$mode" --api_base "$api_base" --max_tokens "$max_tokens" --temperature "$temperature" --threads "$threads" --model_path "$model_path" --port "$port" --gpu "$gpu"
                else
                    python compare.py --input_json "${gen_files[i]}" --ref_json "${gen_files[j]}" --output_json "$comp_file" --model "$model" --mode "$mode" --api_base "$api_base" --max_tokens "$max_tokens" --temperature "$temperature" --threads "$threads" --gpu "$gpu"
                fi
            done
        done
    done
else
    base=$(basename "$input_path")
    gen_files=()
    for gen in "${gen_array[@]}"; do
        gen_files+=("$output_folder/generation/${gen}_${base}")
    done
    num=${#gen_files[@]}
    for (( i=0; i<$num; i++ )); do
        for (( j=i+1; j<$num; j++ )); do
            gen1=$(basename "${gen_files[i]}" | cut -d'_' -f1)
            gen2=$(basename "${gen_files[j]}" | cut -d'_' -f1)
            comp_file="$output_folder/raw_preference_graph_gen/${gen1}_vs_${gen2}_${base}"
            echo "[INFO] Comparing ${gen1} and ${gen2} for $base"
            if [ -n "$model_path" ]; then
                python compare.py --input_json "${gen_files[i]}" --ref_json "${gen_files[j]}" --output_json "$comp_file" --model "$model" --mode "$mode" --api_base "$api_base" --max_tokens "$max_tokens" --temperature "$temperature" --threads "$threads" --model_path "$model_path" --port "$port" --gpu "$gpu"
            else
                python compare.py --input_json "${gen_files[i]}" --ref_json "${gen_files[j]}" --output_json "$comp_file" --model "$model" --mode "$mode" --api_base "$api_base" --max_tokens "$max_tokens" --temperature "$temperature" --threads "$threads" --gpu "$gpu"
            fi
        done
    done
fi

# Step 3: Aggregation
echo "[INFO] Aggregating compare outputs..."
python aggregate.py --input_files "$output_folder/raw_preference_graph_gen" --output_file "$output_folder/aggregation/final_aggregated.json"

echo "[INFO] Pipeline execution completed."
