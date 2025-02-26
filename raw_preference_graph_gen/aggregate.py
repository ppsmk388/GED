#!/usr/bin/env python
"""
aggregate.py

This script aggregates evaluation results from multiple compare JSON files.
For each unique prompt (instruction), it builds a dictionary where the keys are
derived from the compare JSON file names (in the format "generator1 | generator2")
and the values are the final evaluation results (0 or 1).
The final aggregated JSON is saved to the specified output file.
"""

import json
import argparse
import os
import glob

def extract_key_from_filename(filename):
    """
    Extracts a key of the form "generator1 | generator2" from the given filename.
    Assumes the filename follows the pattern: <generator1>_vs_<generator2>_<...>.json
    """
    base = os.path.basename(filename)
    name_no_ext = os.path.splitext(base)[0]
    if "_vs_" in name_no_ext:
        parts = name_no_ext.split("_vs_")
        if len(parts) >= 2:
            gen1 = parts[0]
            # The second part may contain additional underscores; take the first token as generator2.
            gen2 = parts[1].split("_")[0]
            return f"{gen1} | {gen2}"
    return base  # Fallback to the full filename if pattern not met

def aggregate_results(input_files, output_file):
    aggregated = {}
    for file in input_files:
        key = extract_key_from_filename(file)
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                instruction = item.get("instruction", "UNKNOWN")
                value = item.get("final_result")
                if instruction not in aggregated:
                    aggregated[instruction] = {}
                aggregated[instruction][key] = value
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(aggregated, f, indent=4, ensure_ascii=False)
    print(f"[INFO] Aggregated results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate evaluation outputs into a final JSON file.")
    parser.add_argument('--input_files', type=str, required=True,
                        help="Comma-separated list of compare JSON files or a directory containing JSON files")
    parser.add_argument('--output_file', type=str, required=True, help="Final aggregated output JSON file")
    args = parser.parse_args()

    if os.path.isdir(args.input_files):
        files = glob.glob(os.path.join(args.input_files, "*.json"))
    else:
        files = [f.strip() for f in args.input_files.split(",") if f.strip()]
    aggregate_results(files, args.output_file)

if __name__ == "__main__":
    main()
