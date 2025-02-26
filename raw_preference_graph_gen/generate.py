#!/usr/bin/env python
"""
generate.py

This script processes input JSON files to generate model outputs.
For each JSON object, it uses a language model (via vLLM) to generate a response
and appends it as a new field. Additionally, if a --dataset argument is provided,
each item is tagged with that dataset and the dataset value is optionally prepended to the prompt.
"""

import json
import argparse
import os
import logging
from vllm import LLM, SamplingParams

def load_json(file_path):
    """Load JSON data from the specified file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading JSON file {file_path}: {e}")
        raise

def save_json(file_path, data):
    """Save data as JSON to the specified file path."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Error saving JSON file {file_path}: {e}")
        raise

def generate_prompts(json_data, dataset):
    """
    Extract prompts and map each prompt to its corresponding JSON item.
    If a dataset string is provided, it is added to each item and optionally
    prepended to the prompt text.
    Returns a tuple of (list_of_prompts, mapping_dict).
    """
    prompts = []
    mapping = {}
    for item in json_data:
        prompt = item.get("prompt", "")
        prompts.append(prompt)
        mapping[prompt] = item
    return prompts, mapping

def process_json_items(json_data, llm, sampling_params, dataset):
    """
    For each JSON item, generate an output from the model and append it.
    """
    new_json_data = []
    prompts, prompt_item_map = generate_prompts(json_data, dataset)
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        # Retrieve the original item and update its output field
        item = prompt_item_map.get(prompt)
        if item is not None:
            item["output"] = generated_text
            item["dataset"] = dataset
            new_json_data.append(item)
    return new_json_data

def main():
    parser = argparse.ArgumentParser(
        description='Process JSON data with a language model to generate outputs.')
    parser.add_argument('--input_json_path', type=str, required=True,
                        help='Input JSON file path or directory containing JSON files.')
    parser.add_argument('--output_json_path', type=str, required=True,
                        help='Output JSON file path or directory for generated files.')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-0125",
                        help='Model to use for generation.')
    parser.add_argument('--gpu', type=int, default=4, help='Number of GPUs to use.')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for sampling.')
    parser.add_argument('--top_p', type=float, default=0.8, help='Top p sampling parameter.')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens for generation.')
    parser.add_argument('--dataset', type=str, default="", help='Dataset tag to assign to each item.')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Determine if the input path is a directory or a file
    input_files = []
    if os.path.isdir(args.input_json_path):
        for file in os.listdir(args.input_json_path):
            if file.endswith(".json"):
                input_files.append(os.path.join(args.input_json_path, file))
    else:
        input_files.append(args.input_json_path)

    # Prepare output file paths
    output_files = []
    if os.path.isdir(args.output_json_path):
        os.makedirs(args.output_json_path, exist_ok=True)
        for input_file in input_files:
            basename = os.path.basename(input_file)
            output_files.append(os.path.join(args.output_json_path, basename))
    else:
        if len(input_files) > 1:
            logging.error("Multiple input files provided but a single output file specified.")
            exit(1)
        output_files = [args.output_json_path]

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    llm = LLM(model=args.model, gpu_memory_utilization=0.8, tensor_parallel_size=args.gpu)

    for input_file, output_file in zip(input_files, output_files):
        logging.info(f"Processing {input_file} -> {output_file}")
        json_data = load_json(input_file)
        processed_data = process_json_items(json_data, llm, sampling_params, args.dataset)
        save_json(output_file, processed_data)
        logging.info(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    main()