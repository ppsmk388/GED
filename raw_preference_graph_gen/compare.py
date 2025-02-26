#!/usr/bin/env python
"""
compare.py

This script compares outputs from two JSON files (input and reference) using
dataset-specific prompt templates. It builds a chat prompt (with a system message)
and sends it via the chat_completion helper (from utils). If a dataset-specific template
is available (e.g. "HumanEval", "MATH", etc.), it is used; otherwise a fallback prompt is applied.

Usage:
  python compare.py --input_json INPUT.json --ref_json REF.json --output_json OUTPUT.json
         --model <model_name> --mode <random|output_first|reference_first>
         [--api_base <api_base_url>] [--model_path <path>] [--port <port>] [--gpu <num>] [--max_tokens <num>] [--temperature <num>] [--threads <num>]
"""

import json
import argparse
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import re
from utils import chat_completion, start_vllm_server, stop_vllm_server

# Dataset-specific prompt templates
DATASET_TEMPLATES = {
    "humaneval": {
        "system": (
            "You are an expert programmer and code reviewer. Your task is to evaluate code solutions for programming problems. "
            "Assess each solution based on its correctness, efficiency, readability, and adherence to best coding practices."
        ),
        "user": (
            "Please compare the following two code solutions to the given programming problem. For each solution, evaluate whether it produces correct outputs for all edge cases, whether it is efficient in terms of time and space complexity, and whether the code is clean, well-documented, and follows best practices. Identify any errors or areas for improvement.\n\n"
            "Programming Problem:\n{instruction}\n\n"
            "Solution A:\n{output1}\n\n"
            "Solution B:\n{output2}\n\n"
            "Which solution is better and why? Provide a detailed comparison focusing on correctness, efficiency, readability, and coding standards."
        )
    },
    "alpacaeval": {
        "system": (
            "You are an AI assistant trained to assess and compare responses to user instructions. Your evaluations should be based on accuracy, clarity, completeness, and helpfulness."
        ),
        "user": (
            "Please compare the following two responses to the given instruction. Analyze each response for how well it follows the instruction, the accuracy of the information provided, the clarity of the explanation, and the overall helpfulness to the user. Point out any errors, omissions, or areas where the response could be improved.\n\n"
            "Instruction:\n{instruction}\n\n"
            "Response A:\n{output1}\n\n"
            "Response B:\n{output2}\n\n"
            "Which response better addresses the instruction and why? Provide a detailed comparison focusing on the criteria mentioned above."
        )
    },
    "math": {
        "system": (
            "You are a mathematician and educator skilled at evaluating mathematical solutions. Assess the correctness, completeness, and clarity of the following solutions to the math problem. Pay attention to the logical reasoning steps, the mathematical accuracy, and the clarity of explanations."
        ),
        "user": (
            "Please evaluate the following two solutions to the given math problem. For each solution, analyze whether the reasoning is correct, if all necessary steps are included, and if the explanations are clear and easy to understand. Identify any errors or misconceptions.\n\n"
            "Math Problem:\n{instruction}\n\n"
            "Solution A:\n{output1}\n\n"
            "Solution B:\n{output2}\n\n"
            "Which solution is better and why? Provide a detailed comparison focusing on correctness, completeness, and clarity."
        )
    },
    "gsm8k": {
        "system": (
            "You are a teacher specializing in elementary mathematics. Evaluate student answers to math word problems for correctness and quality of reasoning. Consider whether the student has correctly understood the problem, applied appropriate mathematical operations, and provided clear explanations for each step."
        ),
        "user": (
            "Please compare the following two answers to the given math word problem. For each answer, assess the accuracy of the solution, the appropriateness of the reasoning steps, and the clarity of the explanations. Highlight any mistakes or areas for improvement.\n\n"
            "Math Word Problem:\n{instruction}\n\n"
            "Answer A:\n{output1}\n\n"
            "Answer B:\n{output2}\n\n"
            "Which answer is more accurate and better explained, and why? Provide a detailed comparison focusing on the criteria mentioned above."
        )
    },
    "gaia": {
        "system": (
            "You are an expert in complex problem-solving and knowledge retrieval. Assess the following answers for accuracy, relevance, depth, and comprehensiveness in response to the complex question. Consider whether the answers provide correct information, cover all aspects of the question, and are well-articulated."
        ),
        "user": (
            "Please evaluate the following two answers to the given question. For each answer, analyze the correctness of the information provided, the relevance to the question asked, the depth of the explanation, and the overall quality of the response. Note any inaccuracies, omissions, or areas where the answer could be improved.\n\n"
            "Question:\n{instruction}\n\n"
            "Answer A:\n{output1}\n\n"
            "Answer B:\n{output2}\n\n"
            "Which answer provides a better response to the question and why? Provide a detailed comparison focusing on the criteria mentioned above."
        )
    },
    "ultrafeedback": {
        "system": (
            "You are a highly skilled AI assistant trained to evaluate and compare responses to user instructions. Your evaluations should focus on helpfulness, harmlessness, and relevance."
        ),
        "user": (
            "Please compare the following two responses to the given instruction. For each response, assess the following aspects:\n\n"
            "Helpfulness: Does the response effectively address the instruction and provide useful, accurate information?\n"
            "Harmlessness: Does the response avoid any harmful, offensive, or inappropriate content?\n"
            "Relevance: Is the response directly related to the instruction without unnecessary or irrelevant information?\n\n"
            "Instruction:\n{instruction}\n\n"
            "Response A:\n{output1}\n\n"
            "Response B:\n{output2}\n\n"
            "Which response better satisfies the criteria above and why? Provide a detailed explanation supporting your choice, focusing on helpfulness, harmlessness, and relevance."
        )
    }
}

# Fallback prompt (if dataset not found)
FALLBACK_TEMPLATE = {
    "system": (
        "You are a highly efficient assistant who evaluates and selects the best large language model based on the quality of their responses."
    ),
    "user": (
        "I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. "
        "Evaluate the following:\n\n"
        "Instruction:\n{instruction}\n\n"
        "Output A:\n{output1}\n\n"
        "Output B:\n{output2}\n\n"
        "Which output is better and why? Answer with either 'model1' or 'model2' only."
    )
}

def build_messages(instruction, output1, output2, switch, dataset):
    """
    Build chat messages (system and user) based on dataset-specific templates.
    If switch is True, swap output1 and output2.
    """
    candidate_a = output2 if switch else output1
    candidate_b = output1 if switch else output2
    ds_key = dataset.lower() if dataset else ""
    if ds_key in DATASET_TEMPLATES:
        system_prompt = DATASET_TEMPLATES[ds_key]["system"]
        user_prompt = DATASET_TEMPLATES[ds_key]["user"].format(
            instruction=instruction, output1=candidate_a, output2=candidate_b
        )
    else:
        system_prompt = FALLBACK_TEMPLATE["system"]
        user_prompt = FALLBACK_TEMPLATE["user"].format(
            instruction=instruction, output1=candidate_a, output2=candidate_b
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def extract_model_identifier(text, switch):
    """
    Extract the model identifier ("model1" or "model2") from the response text.
    If switch is True, swap the identifier.
    """
    change_back = {"model1": "model2", "model2": "model1"}
    text_segment = text[-50:].lower() if len(text) > 50 else text.lower()
    pattern = r'\b(model1|model2)\b'
    matches = re.findall(pattern, text_segment)
    if matches:
        selected = matches[-1]
    else:
        text_segment_front = text[:50].lower() if len(text) > 50 else text.lower()
        matches = re.findall(pattern, text_segment_front)
        if matches:
            selected = matches[0]
        else:
            return False, None
    if switch:
        selected = change_back[selected]
    return True, selected

def process_item(item, ref_item, mode, api_base, model_name, max_tokens, temperature):
    """
    Process a single evaluation item by building messages and calling chat_completion.
    Returns an evaluation dictionary with the extracted result.
    """
    output1 = item["output"]
    instruction = item["prompt"]
    dataset = item.get("dataset", "")
    output2 = ref_item.get("output")
    # Determine switch based on mode
    if mode == "random":
        switch = random.choice([True, False])
    elif mode == "output_first":
        switch = False
    elif mode == "reference_first":
        switch = True
    else:
        switch = random.choice([True, False])
    
    messages = build_messages(instruction, output1, output2, switch, dataset)
    # Call chat_completion from utils (which uses the OpenAI API interface)
    try:
        response = chat_completion(api_base=api_base, model_name=model_name,
                                   messages=messages, max_tokens=max_tokens, temperature=temperature)
    except Exception as e:
        response = f"[ERROR] {str(e)}"
    success, result = extract_model_identifier(response, switch)
    eval_item = {
        "instruction": instruction,
        "dataset": dataset,
        "output_1": output1,
        "output_2": output2,
        "result": result if success else "[ERROR]",
        "full_response": response
    }
    return eval_item

def process_tasks(input_json_path, ref_json_path, output_json_path, api_base, model_name, mode, max_tokens, temperature, threads):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    with open(ref_json_path, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    
    # Create mapping for reference items by (prompt, dataset)
    ref_mapping = {(item["prompt"], item.get("dataset", "")): item for item in ref_data}
    
    results = []
    tasks = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for item in input_data:
            key = (item["prompt"], item.get("dataset", ""))
            if key in ref_mapping:
                ref_item = ref_mapping[key]
                tasks.append(executor.submit(process_item, item, ref_item, mode, api_base, model_name, max_tokens, temperature))
        for future in as_completed(tasks):
            results.append(future.result())
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved evaluation results to {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description='Run evaluation with specified configuration and model using chat_completion.')
    parser.add_argument('--input_json', type=str, required=True, help='Input JSON file')
    parser.add_argument('--ref_json', type=str, required=True, help='Reference JSON file')
    parser.add_argument('--output_json', type=str, required=True, help='Output JSON file')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--mode', type=str, required=True, choices=["random", "output_first", "reference_first"],
                        help='Evaluation mode')
    parser.add_argument('--api_base', type=str, required=True, help='Base URL for the API (e.g., http://localhost:8000)')
    parser.add_argument('--max_tokens', type=int, default=256, help='Maximum tokens for evaluation response')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for evaluation')
    parser.add_argument('--threads', type=int, default=10, help='Number of concurrent threads')
    # Optional parameters for starting a vLLM server
    parser.add_argument('--model_path', type=str, default=None, help='Path to the model to be hosted by vLLM server')
    parser.add_argument('--port', type=int, default=8000, help='Port to host the vLLM server on')
    parser.add_argument('--gpu', type=int, default=1, help='Number of GPUs to use for the vLLM server')
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    
    server_process = None
    if args.model_path:
        logging.info(f"Starting vLLM server for model {args.model_path} on port {args.port} using {args.gpu} GPU(s).")
        server_process = start_vllm_server(args.model_path, args.model, args.port, args.gpu)
        # Update api_base to include port if needed
        if f":{args.port}" not in args.api_base:
            args.api_base = f"http://localhost:{args.port}"
    
    process_tasks(args.input_json, args.ref_json, args.output_json, args.api_base, args.model,
                  args.mode, args.max_tokens, args.temperature, args.threads)
    
    if server_process:
        stop_vllm_server(server_process)

if __name__ == "__main__":
    main()
