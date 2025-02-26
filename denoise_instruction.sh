#!/bin/bash

# Usage: ./run_response_rank.sh --eval_model <model> --answer_model <model> --task_name <task> --w_type <type> --rank_type <type>

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --eval_model) eval_model="$2"; shift ;;
        --answer_model) answer_model="$2"; shift ;;
        --task_name) task_name="$2"; shift ;;
        --w_type) w_type="$2"; shift ;;
        --rank_type) rank_type="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if all required parameters are provided
if [[ -z "$eval_model" || -z "$answer_model" || -z "$task_name" || -z "$w_type" || -z "$rank_type" ]]; then
    echo "Usage: ./run_response_rank.sh --eval_model <model> --answer_model <model> --task_name <task> --w_type <type> --rank_type <type>"
    exit 1
fi

# Run the Python script with the provided arguments
python response_rank_gen.py \
    --eval_model "$eval_model" \
    --answer_model "$answer_model" \
    --task_name "$task_name" \
    --w_type "$w_type" \
    --rank_type "$rank_type"
