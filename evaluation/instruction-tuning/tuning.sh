#!/bin/bash


# Exit immediately if a command exits with a non-zero status
set -e


# Initialize variables
MODEL_NAME_OR_PATH=""
DATASET=""
DATASET_DIR=""
OUTPUT_DIR=""

# Define base paths
BASE_MODEL_PATH=""
BASE_DATASET_DIR=""
BASE_OUTPUT_DIR=""
BASE_SCRIPT_DIR=""
BASE_JSON_DIR=""

BASE_MODEL_PATH="your base model path"
BASE_DATASET_DIR="your base dataset path"
BASE_OUTPUT_DIR="your base output path"
BASE_SCRIPT_DIR="xxxx/LLaMA-Factory/src/train.py"
BASE_JSON_DIR="xxxx/LLaMA-Factory/examples/deepspeed/ds_z3_config.json"



echo "BASE_DATASET_DIR: $BASE_DATASET_DIR"

# Parse command-line options using getopts
while getopts ":m:d:s:o:" opt; do
  case $opt in
    m) MODEL_NAME_OR_PATH="$OPTARG" ;;
    d) DATASET="$OPTARG" ;;
    s) DATASET_DIR="$OPTARG" ;;
    o) OUTPUT_DIR="$OPTARG" ;;
    \?) echo "Invalid option: -$OPTARG" >&2
        exit 1 ;;
  esac
done

# Use default paths if not specified via command-line arguments
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-$BASE_MODEL_PATH}"
DATASET_DIR="${DATASET_DIR:-$BASE_DATASET_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE_OUTPUT_DIR}"


DATASETS=("tuning dataset name" )



# Loop over DATASETS
for CURRENT_DATASET in "${DATASETS[@]}"; do
  # Construct OUTPUT_DIR based on CURRENT_DATASET
  CURRENT_OUTPUT_DIR="${BASE_OUTPUT_DIR}/${CURRENT_DATASET}"

  # Call the function to find the latest checkpoint
  OUTPUT_DIR="$CURRENT_OUTPUT_DIR"  # Update OUTPUT_DIR for the function


  # Print the CURRENT_DATASET and RESUME_FROM_CHECKPOINT for debugging
  echo "Processing DATASET: $CURRENT_DATASET"
  echo "OUTPUT_DIR: $CURRENT_OUTPUT_DIR"

  echo "$CURRENT_OUTPUT_DIR"

  deepspeed --num_gpus 8 $BASE_SCRIPT_DIR \
    --deepspeed $BASE_JSON_DIR \
    --stage sft \
    --do_train \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --dataset "$CURRENT_DATASET" \
    --dataset_dir "$DATASET_DIR" \
    --template llama2 \
    --finetuning_type full \
    --output_dir "$CURRENT_OUTPUT_DIR" \
    --overwrite_cache \
    --report_to tensorboard \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 2000 \
    --eval_steps 10000 \
    --evaluation_strategy steps \
    --learning_rate 1.0e-5 \
    --num_train_epochs 3 \
    --val_size 0.1 \
    --ddp_timeout 1800000 \
    --plot_loss \
    --fp16
done

