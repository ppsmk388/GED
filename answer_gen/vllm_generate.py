import json
import argparse
from vllm import LLM, SamplingParams
import os

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
    

prompt_to_instruct = {}

def generate_prompt(prompt, item):
    prompt_to_instruct[prompt] = item
    return prompt


def decode_get_item(prompt):
    return prompt_to_instruct[prompt]



def process_json_items(json_data, llm, sampling_params):
    new_json_data = []


    prompts = [generate_prompt(item["prompt"], item) for item in json_data]
    
    outputs = llm.generate(prompts, sampling_params)

    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        this_item = decode_get_item(prompt)
        this_item["output"] = generated_text
        new_json_data.append(this_item)
    return new_json_data
        

def save_json(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Process JSON data with a language model.')
    parser.add_argument('--input_json_path', type=str,default="./ultrafeedback_sample_5k.json", help='Input JSON file path')
    parser.add_argument('--output_json_path', type=str, required=True, help='Output JSON file path')
    parser.add_argument('--model', type=str, default="gpt-3.5-turbo-0125", help='Model to use for generation')
    parser.add_argument('--gpu', type=int, default=4, help='Number of GPUs to use')


    args = parser.parse_args()
    json_data = load_json(args.input_json_path)

    output_json_path_list = args.output_json_path.split(',')

    
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=2048)


    llm = LLM(model=args.model,gpu_memory_utilization=0.8, tensor_parallel_size=args.gpu)

    
    for output_json_path in output_json_path_list:
        output_json_folder = output_json_path.split('/')
        output_json_folder = '/'.join(output_json_folder[:-1])
        
        if os.path.exists(output_json_folder) == False:
            os.makedirs(output_json_folder)
        
        
        processed_data = process_json_items(json_data, llm, sampling_params)

        save_json(output_json_path, processed_data)

if __name__ == "__main__":
    main()